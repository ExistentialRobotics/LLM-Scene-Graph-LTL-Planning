#include "erl_env/finite_state_automaton.hpp"
#include "erl_env/spot_helper.hpp"

#include <spot/parseaut/public.hh>
#include <spot/twaalgos/complete.hh>

namespace erl::env {

    FiniteStateAutomaton::Setting::Setting(const std::string &filepath, const FileType file_type, const bool complete) {
        switch (file_type) {
            case FileType::kYaml:
                FromYamlFile(filepath);
                break;
            case FileType::kSpotHoa:
                FromSpotGraphHoaFile(filepath, complete);
                break;
            case FileType::kBoostDot:
                FromBoostGraphDotFile(filepath);
                break;
        }
    }

    FiniteStateAutomaton::BoostGraph
    FiniteStateAutomaton::Setting::AsBoostGraph() const {
        std::stringstream ss;
        ss << atomic_propositions[0];
        const std::size_t num_aps = atomic_propositions.size();
        for (std::size_t i = 1; i < num_aps; ++i) { ss << " " << atomic_propositions[i]; }
        BoostGraph graph(0, BoostGraphProp(ss.str()));
        // add vertex with label
        std::vector<boost::graph_traits<BoostGraph>::vertex_descriptor> vertices;
        vertices.reserve(num_states);
        for (uint32_t state = 0; state < num_states; ++state) {
            std::string label = std::to_string(state);
            if (state == initial_state) {
                label += ":initial";
            } else if (std::find(accepting_states.begin(), accepting_states.end(), state) != accepting_states.end()) {
                label += ":accepting";
            }
            vertices.emplace_back(boost::add_vertex(BoostVertexProp(state, {std::to_string(state), label}), graph));
        }
        // add edge with label
        for (auto &transition: transitions) {
            for (auto &label: transition.labels) { boost::add_edge(vertices[transition.from], vertices[transition.to], BoostEdgeProp(label), graph); }
        }
        return graph;
    }

    void
    FiniteStateAutomaton::Setting::AsBoostGraphDotFile(const std::string &filename) const {
        std::ofstream ofs(filename);
        std::map<std::string, std::string> graph_attr, vertex_attr, edge_attr;
        auto graph = AsBoostGraph();
        graph_attr["name"] = boost::get_property(graph, boost::graph_name);
        auto vertex_label_map = boost::get(boost::vertex_color, graph);
        auto edge_label_map = boost::get(boost::edge_color, graph);
        boost::write_graphviz(
            ofs,
            graph,
            boost::make_label_writer(vertex_label_map),
            boost::make_label_writer(edge_label_map),
            boost::make_graph_attributes_writer(graph_attr, vertex_attr, edge_attr));
    }

    void
    FiniteStateAutomaton::Setting::FromBoostGraphDotFile(const std::string &filepath) {
        std::ifstream dot_file(filepath);
        ERL_ASSERTM(dot_file.is_open(), "failed to open the dot file");
        BoostGraph graph(0);
        boost::dynamic_properties properties;
        // use ref_property_map to turn a graph property into a property map
        boost::ref_property_map<BoostGraph *, std::string> graph_name = boost::get_property(graph, boost::graph_name);
        boost::property_map<BoostGraph, boost::vertex_index_t>::type vertex_index = boost::get(boost::vertex_index, graph);
        boost::property_map<BoostGraph, boost::vertex_name_t>::type vertex_name = boost::get(boost::vertex_name, graph);
        boost::property_map<BoostGraph, boost::vertex_color_t>::type vertex_label = boost::get(boost::vertex_color, graph);
        boost::property_map<BoostGraph, boost::edge_color_t>::type edge_label = boost::get(boost::edge_color, graph);
        properties.property("name", graph_name);
        properties.property("node_index", vertex_index);
        properties.property("node_id", vertex_name);
        properties.property("label", vertex_label);
        properties.property("label", edge_label);
        boost::read_graphviz(dot_file, graph, properties);

        num_states = boost::num_vertices(graph);
        // get atomic propositions from graph name
        std::string graph_name_str = graph_name[&graph];
        std::stringstream ss(graph_name_str);
        std::string token;
        while (std::getline(ss, token, ' ')) { atomic_propositions.emplace_back(token); }
        // iterate over vertices
        bool initial_state_found = false;
        for (auto v = boost::vertices(graph); v.first != v.second; ++v.first) {
            uint32_t state = vertex_index[*v.first];
            if (std::string label = vertex_label[*v.first]; label.find("initial") != std::string::npos) {
                ERL_ASSERTM(!initial_state_found, "multiple initial states");
                initial_state = state;
                initial_state_found = true;
            } else if (label.find("accepting") != std::string::npos) {
                accepting_states.emplace_back(state);
            }
        }
        // iterate over edges
        std::vector<std::tuple<uint32_t, uint32_t, std::set<uint32_t>>> loaded_transitions(num_states * num_states);
        for (auto e = boost::edges(graph); e.first != e.second; ++e.first) {
            uint32_t from = vertex_index[boost::source(*e.first, graph)];
            uint32_t to = vertex_index[boost::target(*e.first, graph)];
            uint32_t label = edge_label[*e.first];
            std::size_t index = from * num_states + to;
            auto &[from_, to_, labels] = loaded_transitions[index];
            from_ = from;
            to_ = to;
            labels.insert(label);
        }
        // construct transitions
        for (auto &[from, to, labels]: loaded_transitions) {
            if (labels.empty()) { continue; }
            transitions.emplace_back(from, to, labels);
        }
        std::sort(accepting_states.begin(), accepting_states.end(), std::greater<>());
        for (auto &transition: transitions) { std::sort(transition.labels.begin(), transition.labels.end(), std::greater<>()); }
    }

    FiniteStateAutomaton::SpotGraph
    FiniteStateAutomaton::Setting::AsSpotGraph() const {
        // https://spot.lre.epita.fr/tut22.html
        const spot::bdd_dict_ptr dict = spot::make_bdd_dict();
        SpotGraph graph = spot::make_twa_graph(dict);  // Buchi automaton

        std::vector<bdd> bdd_aps;
        bdd_aps.reserve(bdd_aps.size());
        for (auto &ap: atomic_propositions) {
            const int ap_index = graph->register_ap(ap);
            ERL_ASSERTM(ap_index >= 0, "spot error: ap_index < 0");
            bdd ap_bdd = bdd_ithvar(ap_index);
            bdd_aps.emplace_back(ap_bdd);
        }

        // we do not support multiple accepting sets yet: https://spot.lre.epita.fr/concepts.html#acceptance-set
        graph->set_generalized_buchi(1);       // set the number of acceptance sets to use
        graph->new_states(num_states);         // set the number of states
        graph->set_init_state(initial_state);  // set the initial state
        for (auto &transition: transitions) {
            bdd cond = bddfalse;
            for (auto &label: transition.labels) { cond |= spot_helper::LabelToBdd(label, bdd_aps); }
            if (std::find(accepting_states.begin(), accepting_states.end(), transition.to) != accepting_states.end()) {
                graph->new_edge(transition.from, transition.to, cond, {0});
            } else {
                graph->new_edge(transition.from, transition.to, cond);
            }
        }

        return graph;
    }

    void
    FiniteStateAutomaton::Setting::FromSpotGraphHoaFile(const std::string &filepath, const bool complete) {
        spot::parsed_aut_ptr pa = spot::parse_aut(filepath, spot::make_bdd_dict());
        ERL_ASSERTM(pa != nullptr, "failed to parse the HOA file");
        ERL_ASSERTM(!pa->format_errors(std::cerr), "HOA format error");
        ERL_ASSERTM(!pa->aborted, "HOA parsing aborted");
        ERL_ASSERTM(pa->aut != nullptr, "spot error: pa->aut is nullptr");
        ERL_ASSERTM(pa->aut->num_sets() == 1, "only support single set of accepting states.");

        if (complete) { spot::complete_here(pa->aut); }  // complete the automaton

        const spot::bdd_dict_ptr &bdd_dict = pa->aut->get_dict();

        num_states = pa->aut->num_states();
        initial_state = pa->aut->get_init_state_number();
        atomic_propositions.resize(pa->aut->ap().size());
        // extract atomic propositions
        std::vector<bdd> bdd_aps;
        bdd_aps.reserve(pa->aut->ap().size());
        bdd ap_vars = bddtrue;
        for (const spot::formula &ap: pa->aut->ap()) {
            int index = bdd_dict->varnum(ap);
            ERL_ASSERTM(index >= 0, "spot error: index < 0");
            atomic_propositions[index] = ap.ap_name();
            bdd_aps.emplace_back(spot::formula_to_bdd(ap, bdd_dict, this));
            ap_vars &= bdd_aps.back();
        }
        // get sink states, they should not be added to the accepting states
        std::vector<bool> sink_states(num_states);
        for (uint32_t s = 0; s < num_states; ++s) { sink_states[s] = spot_helper::IsSink(pa->aut, s); }
        // extract transitions and accepting states
        std::unordered_set<uint32_t> accepting_set;
        std::vector<std::tuple<uint32_t, uint32_t, std::set<uint32_t>>> loaded_transitions;
        loaded_transitions.resize(num_states * num_states);
        for (uint32_t s = 0; s < num_states; ++s) {
            auto out_edges = pa->aut->out(s);
            for (auto &t: out_edges) {
                // if (t.acc.count() > 0 && !sink_states[t.dst]) { accepting_set.insert(t.dst); }
                if (t.src == t.dst && t.acc.count() > 0) { accepting_set.insert(t.dst); }  // accepting state has self loop with acceptance condition
                ERL_ASSERTM(spot::bdd_to_formula(t.cond, bdd_dict).is_ltl_formula(), "Only support LTL formula.");
                auto &[from, to, labels] = loaded_transitions[s * num_states + t.dst];
                from = t.src;
                to = t.dst;
                std::vector<uint32_t> new_labels = spot_helper::BddToLabels(t.cond, ap_vars);
                for (auto &label: new_labels) { labels.insert(label); }
            }
        }
        bdd_dict->unregister_all_my_variables(this);  // unregister all variables to please spot library
        accepting_states.insert(accepting_states.end(), accepting_set.begin(), accepting_set.end());
        for (auto &[from, to, labels]: loaded_transitions) {
            if (labels.empty()) { continue; }
            transitions.emplace_back(from, to, labels);
        }
        std::sort(accepting_states.begin(), accepting_states.end(), std::greater<>());
        for (auto &transition: transitions) { std::sort(transition.labels.begin(), transition.labels.end(), std::greater<>()); }
    }

    FiniteStateAutomaton::FiniteStateAutomaton(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr");
        Init();
    }

    void
    FiniteStateAutomaton::Init() {
        m_alphabet_size_ = 1 << m_setting_->atomic_propositions.size();

        // construct transition labels and next state
        ERL_ASSERTM(!m_setting_->transitions.empty(), "transitions is empty");
        for (auto &transition: m_setting_->transitions) {
            {
                uint32_t key = HashingTransition(transition.from, transition.to);
                auto [itr, new_label] = m_transition_labels_.emplace(key, transition.labels);
                ERL_ASSERTM(new_label, "duplicated transition");
            }

            if (transition.from == transition.to) { continue; }

            for (auto &label: transition.labels) {
                uint32_t key = HashingStateLabelPair(transition.from, label);
                auto [itr, new_next_state] = m_transition_next_state_.emplace(key, transition.to);
                ERL_ASSERTM(new_next_state, "duplicated transition");
            }
        }

        // set accepting states
        m_accepting_states_.resize(m_setting_->num_states, false);
        for (const auto &state: m_setting_->accepting_states) { m_accepting_states_[state] = true; }

        // compute levels
        m_levels_.emplace_back(m_setting_->accepting_states.begin(), m_setting_->accepting_states.end());  // level 0
        m_levels_b_.emplace_back(m_setting_->num_states, false);                                           // level 0
        m_sink_states_.resize(m_setting_->num_states, true);
        for (const auto &state: m_setting_->accepting_states) {
            m_sink_states_[state] = false;
            m_levels_b_[0][state] = true;
        }
        uint32_t level = 0;
        while (true) {
            bool done = true;
            auto &level_states = m_levels_[level];
            for (auto &state: level_states) {
                for (uint32_t prev_state = 0; prev_state < m_setting_->num_states; ++prev_state) {
                    if (!m_sink_states_[prev_state]) { continue; }
                    if (m_transition_labels_.find(HashingTransition(prev_state, state)) == m_transition_labels_.end()) { continue; }
                    // exist a transition from prev_state to state but prev_state is marked as a sink state
                    m_sink_states_[prev_state] = false;
                    if (done) {  // add a new level
                        done = false;
                        m_levels_.emplace_back(1, prev_state);
                        m_levels_b_.emplace_back(m_setting_->num_states, false);
                        m_levels_b_[level + 1][prev_state] = true;
                    } else {
                        m_levels_[level + 1].emplace_back(prev_state);
                        m_levels_b_[level + 1][prev_state] = true;
                    }
                }
            }
            if (done) { break; }
            level++;
        }
    }
}  // namespace erl::env
