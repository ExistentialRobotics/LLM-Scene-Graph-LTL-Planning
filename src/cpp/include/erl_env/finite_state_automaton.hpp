#pragma once

#include "erl_common/assert.hpp"
#include "erl_common/yaml.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <absl/container/flat_hash_map.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#pragma GCC diagnostic pop

#include <spot/twaalgos/hoa.hh>
#include <spot/twa/twagraph.hh>
#include <spot/twaalgos/complete.hh>

namespace erl::env {

    /**
     * An interface of Büchi automaton.
     */
    class FiniteStateAutomaton {

    public:
        // clang-format off
        typedef boost::property<boost::vertex_index_t, uint32_t,
                boost::property<boost::vertex_name_t, std::string,
                boost::property<boost::vertex_color_t, std::string>>> BoostVertexProp;
        // clang-format on
        typedef boost::property<boost::edge_color_t, uint32_t> BoostEdgeProp;
        typedef boost::property<boost::graph_name_t, std::string> BoostGraphProp;
        typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, BoostVertexProp, BoostEdgeProp, BoostGraphProp> BoostGraph;
        using SpotGraph = spot::twa_graph_ptr;

        struct Setting final : common::Yamlable<Setting> {

            struct Transition {
                uint32_t from = 0;
                uint32_t to = 0;
                std::vector<uint32_t> labels = {};

                Transition() = default;

                Transition(uint32_t from, uint32_t to, const std::set<uint32_t> &labels)
                    : from(from),
                      to(to),
                      labels(labels.begin(), labels.end()) {}
            };

            uint32_t num_states = 0;                            // number of states
            uint32_t initial_state = 0;                         // initial state
            std::vector<uint32_t> accepting_states = {};        // accepting states
            std::vector<std::string> atomic_propositions = {};  // atomic propositions
            std::vector<Transition> transitions = {};           // transitions

            enum class FileType { kSpotHoa = 0, kBoostDot = 1, kYaml = 2 };

            Setting() = default;
            Setting(const std::string &filepath, FileType file_type, bool complete = false);

            [[nodiscard]] BoostGraph
            AsBoostGraph() const;

            void
            AsBoostGraphDotFile(const std::string &filename) const;

            void
            FromBoostGraphDotFile(const std::string &filepath);

            [[nodiscard]] SpotGraph
            AsSpotGraph() const;

            /**
             * @brief save the automaton as a HOA file.
             * @param filename
             * @param complete if true, the automaton is guaranteed to be complete.
             */
            void
            AsSpotGraphHoaFile(const std::string &filename, const bool complete) const {
                std::ofstream ofs(filename);
                SpotGraph graph = AsSpotGraph();
                if (complete) { spot::complete_here(graph); }
                spot::print_hoa(ofs, graph);
            }

            void
            FromSpotGraphHoaFile(const std::string &filepath, bool complete);

            void
            AsSpotGraphDotFile(const std::string &filename) const {
                std::ofstream ofs(filename);
                AsSpotGraph()->dump_storage_as_dot(ofs);
            }
        };

    protected:
        std::shared_ptr<Setting> m_setting_;                                        // setting
        uint32_t m_alphabet_size_ = 0;                                              // alphabet size
        absl::flat_hash_map<uint32_t, std::vector<uint32_t>> m_transition_labels_;  // given key of p->q, return the edge labels
        absl::flat_hash_map<uint32_t, uint32_t> m_transition_next_state_;           // given hashing of state p and label a, return the next state q
        std::vector<std::vector<uint32_t>> m_levels_;                               // states in each level
        std::vector<std::vector<bool>> m_levels_b_;                                 // states in each level (boolean version)
        std::vector<bool> m_sink_states_;                                           // sink states
        std::vector<bool> m_accepting_states_;                                      // accepting states

    public:
        explicit FiniteStateAutomaton(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] uint32_t
        GetAlphabetSize() const {
            return m_alphabet_size_;
        }

        [[nodiscard]] std::vector<std::vector<uint32_t>>
        GetLevels() const {
            return m_levels_;
        }

        [[nodiscard]] std::vector<std::vector<bool>>
        GetLevelsB() const {
            return m_levels_b_;
        }

        [[nodiscard]] std::vector<bool>
        GetSinkStates() const {
            return m_sink_states_;
        }

        [[nodiscard]] std::vector<uint32_t>
        GetTransitionLabels(const uint32_t from, const uint32_t to) const {
            auto result = m_transition_labels_.find(HashingTransition(from, to));
            if (result == m_transition_labels_.end()) { return {}; }
            return result->second;
        }

        [[nodiscard]] uint32_t
        GetNextState(const uint32_t state, const uint32_t label) const {
            const auto result = m_transition_next_state_.find(HashingStateLabelPair(state, label));
            if (result == m_transition_next_state_.end()) { return state; }
            return result->second;
        }

        /**
         * @brief Get the lowest level index of the given state.
         * @param state
         * @return
         */
        [[nodiscard]] int
        GetLevelIndex(uint32_t state) const {
            if (m_sink_states_[state]) { return -1; }
            for (std::size_t i = 0; i < m_levels_.size(); ++i) {
                if (m_levels_b_[i][state]) { return static_cast<int>(i); }
            }
            throw std::runtime_error("state is not in any level and is not a sink state!");
        }

        [[nodiscard]] bool
        IsSinkState(const uint32_t state) const {
            return m_sink_states_[state];
        }

        [[nodiscard]] bool
        IsAcceptingState(const uint32_t state) const {
            return m_accepting_states_[state];
        }

    private:
        void
        Init();

        [[nodiscard]] uint32_t
        HashingTransition(const uint32_t from, const uint32_t to) const {
            return from + to * m_setting_->num_states;  // column-major
        }

        [[nodiscard]] uint32_t
        HashingStateLabelPair(const uint32_t state, const uint32_t label) const {
            return state + label * m_setting_->num_states;  // column-major
        }
    };

}  // namespace erl::env

namespace YAML {

    template<>
    struct convert<erl::env::FiniteStateAutomaton::Setting::Transition> {
        static Node
        encode(const erl::env::FiniteStateAutomaton::Setting::Transition &rhs) {
            Node node(NodeType::Sequence);
            node.push_back(Node(std::vector{rhs.from, rhs.to}));
            node.push_back(rhs.labels);
            return node;
        }

        static bool
        decode(const Node &node, erl::env::FiniteStateAutomaton::Setting::Transition &rhs) {
            ERL_ASSERTM(node.IsSequence(), "node is not a sequence");
            rhs.from = node[0][0].as<uint32_t>();
            rhs.to = node[0][1].as<uint32_t>();
            rhs.labels = node[1].as<std::vector<uint32_t>>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::env::FiniteStateAutomaton::Setting::Transition &rhs) {
        out << Flow << BeginSeq << Flow;
        out << BeginSeq << rhs.from << rhs.to << EndSeq;
        out << Flow << rhs.labels;
        out << EndSeq;
        return out;
    }

    template<>
    struct convert<erl::env::FiniteStateAutomaton::Setting> {
        static Node
        encode(const erl::env::FiniteStateAutomaton::Setting &rhs) {
            Node node;
            node["num_states"] = rhs.num_states;
            node["initial_state"] = rhs.initial_state;
            node["accepting_states"] = rhs.accepting_states;
            node["atomic_propositions"] = rhs.atomic_propositions;
            node["transitions"] = rhs.transitions;
            return node;
        }

        static bool
        decode(const Node &node, erl::env::FiniteStateAutomaton::Setting &rhs) {
            rhs.num_states = node["num_states"].as<uint32_t>();
            rhs.initial_state = node["initial_state"].as<uint32_t>();
            rhs.accepting_states = node["accepting_states"].as<std::vector<uint32_t>>();
            rhs.atomic_propositions = node["atomic_propositions"].as<std::vector<std::string>>();
            rhs.transitions = node["transitions"].as<std::vector<erl::env::FiniteStateAutomaton::Setting::Transition>>();
            std::sort(rhs.accepting_states.begin(), rhs.accepting_states.end(), std::greater<>());
            for (auto &transition: rhs.transitions) { std::sort(transition.labels.begin(), transition.labels.end(), std::greater<>()); }
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::env::FiniteStateAutomaton::Setting &rhs) {
        out << BeginMap;
        out << Key << "num_states" << Value << rhs.num_states;
        out << Key << "initial_state" << Value << rhs.initial_state;
        out << Key << "accepting_states" << Value << Flow << rhs.accepting_states;
        out << Key << "atomic_propositions" << Value << Flow << rhs.atomic_propositions;
        out << Key << "transitions" << Value << rhs.transitions;
        out << EndMap;
        return out;
    }
}  // namespace YAML
