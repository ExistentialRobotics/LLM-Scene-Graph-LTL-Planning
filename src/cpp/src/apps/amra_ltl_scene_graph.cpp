#include <boost/program_options.hpp>
#include <filesystem>
#include "erl_search_planning/amra_star.hpp"
#include "erl_search_planning/planning_interface_multi_resolutions.hpp"
#include "erl_search_planning/ltl_3d_heuristic.hpp"
#include "erl_search_planning/llm_scene_graph_heuristic.hpp"
#include "erl_env/environment_ltl_scene_graph.hpp"
#include "erl_common/yaml.hpp"

struct Options final : erl::common::Yamlable<Options> {
    std::string output_dir = {};
    std::string scene_graph_file = {};
    std::string map_data_dir = {};
    std::string automaton_file = {};
    bool make_automaton_complete = false;  // make the automaton complete
    std::string ap_file = {};
    std::string llm_heuristic_file = {};
    int init_grid_x = -1;
    int init_grid_y = -1;
    int init_grid_z = -1;
    double robot_radius = 0.0;
    double object_reach_radius = 0.6;
    erl::env::scene_graph::Node::Type max_level = erl::env::scene_graph::Node::Type::kFloor;
    std::string ltl_heuristic_layout = {};
    std::string llm_heuristic_layout = {};
    int repeat = 1;
    bool save_amra_log = false;
    bool hold_for_visualization = false;
};

inline static const auto *hierarchy_layout = R"(Hierarchical Planning Domain Layout:
|  LEVEL  | Anchor | Occupancy | Object | Room | Floor |
| ENABLED |   %c    |     %c     |   %c    |  %c   |   %c   |
|   LTL   |   %c    |     %c     |   %c    |  %c   |   %c   |
|   LLM   |   %c    |     %c     |   %c    |  %c   |   %c   |)";

int
main(int argc, char *argv[]) {
    using namespace erl::env;
    using namespace erl::search_planning;

    std::string max_level_str;
    Options options;
    bool use_llm_heuristic;
    char level_enabled_chars[5] = {'1', '0', '0', '0', '0'};
    char level_ltl_chars[5] = {'1', 'X', 'X', 'X', 'X'};
    char level_llm_chars[5] = {'X', 'X', 'X', 'X', 'X'};

    try {
        namespace po = boost::program_options;
        po::options_description desc;
        po::positional_options_description positional_options;
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            ("output-dir", po::value<std::string>(&options.output_dir), "path to save the results.")
            ("scene-graph-file", po::value<std::string>(&options.scene_graph_file), "path to the YAML of scene graph.")
            ("map-data-dir", po::value<std::string>(&options.map_data_dir), "path to the map data directory.")
            ("automaton-file", po::value<std::string>(&options.automaton_file), "path to the automaton file.")
            ("make-automaton-complete", po::bool_switch(&options.make_automaton_complete)->default_value(options.make_automaton_complete), "make the automaton complete.")
            ("ap-file", po::value<std::string>(&options.ap_file), "path to the AP description file.")
            ("llm-heuristic-file", po::value<std::string>(&options.llm_heuristic_file), "path to the LLM heuristic file.")
            ("init-grid-x", po::value<int>(&options.init_grid_x), "initial x position in grids.")
            ("init-grid-y", po::value<int>(&options.init_grid_y), "initial y position in grids.")
            ("init-grid-z", po::value<int>(&options.init_grid_z), "initial z position in grids.")
            ("robot-radius", po::value<double>(&options.robot_radius)->default_value(options.robot_radius), "robot radius.")
            ("object-reach-radius", po::value<double>(&options.object_reach_radius)->default_value(options.object_reach_radius), "object reach radius.")
            ("max-level", po::value<std::string>(&max_level_str), "maximum level to plan: kOcc, kObject, kRoom, kFloor, anchor cannot be disabled.")
            ("ltl-heuristic-config", po::value<std::string>(&options.ltl_heuristic_layout), "a sequence of 0,1 to indicate whether to use LTL heuristic for each level up to the max_level: anchor, kOcc, kObject, kRoom, kFloor.")
            ("llm-heuristic-config", po::value<std::string>(&options.llm_heuristic_layout), "a sequence of 0,1 to indicate whether to use LLM heuristic for each level up to the max_level: anchor, kOcc, kObject, kRoom, kFloor.")
            ("repeat", po::value<int>(&options.repeat)->default_value(options.repeat), "repeat the experiment for multiple times.")
            ("save-amra-log", po::bool_switch(&options.save_amra_log)->default_value(options.save_amra_log), "save AMRA* log")
            ("hold-for-visualization", po::bool_switch(&options.hold_for_visualization)->default_value(options.hold_for_visualization), "pause for visualization");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options] amra_ltl_scene_graph" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(vm);
        if (options.output_dir.empty()) {
            std::cerr << "--output-dir is not provided." << std::endl;
            return 1;
        }
        if (options.scene_graph_file.empty()) {
            std::cerr << "--scene-graph-file is not provided." << std::endl;
            return 1;
        }
        if (options.map_data_dir.empty()) {
            std::cerr << "--map-data-dir is not provided." << std::endl;
            return 1;
        }
        if (options.automaton_file.empty()) {
            std::cerr << "--automaton-file is not provided." << std::endl;
            return 1;
        }
        if (options.ap_file.empty()) {
            std::cerr << "--ap-file is not provided." << std::endl;
            return 1;
        }
        if (options.init_grid_x < 0) {
            std::cerr << "--init-grid-x is not provided." << std::endl;
            return 1;
        }
        if (options.init_grid_y < 0) {
            std::cerr << "--init-grid-y is not provided." << std::endl;
            return 1;
        }
        if (options.init_grid_z < 0) {
            std::cerr << "--init-grid-z is not provided." << std::endl;
            return 1;
        }
        if (options.object_reach_radius < options.robot_radius) {
            std::cerr << "--object-reach-radius should be at least robot_radius." << std::endl;
            return 1;
        }

        if (max_level_str.empty()) {
            std::cerr << "--max-level is not provided." << std::endl;
            return 1;
        }
        if (max_level_str == "kOcc") {
            options.max_level = scene_graph::Node::Type::kOcc;
        } else if (max_level_str == "kFloor") {
            options.max_level = scene_graph::Node::Type::kFloor;
        } else if (max_level_str == "kRoom") {
            options.max_level = scene_graph::Node::Type::kRoom;
        } else if (max_level_str == "kObject") {
            options.max_level = scene_graph::Node::Type::kObject;
        } else {
            std::cerr << "--max-level is not valid." << std::endl;
            return 1;
        }
        std::memset(level_enabled_chars + 1, '1', static_cast<std::size_t>(options.max_level) + 1);

        if (options.ltl_heuristic_layout.empty()) {
            std::cerr << "--ltl-heuristic-config is not provided." << std::endl;
            return 1;
        }
        if (options.ltl_heuristic_layout.length() != static_cast<std::size_t>(options.max_level) + 2) {
            std::cerr << "--ltl-heuristic-config is not valid with max_level = " << max_level_str << std::endl;
            return 1;
        }
        if (std::any_of(options.ltl_heuristic_layout.begin(), options.ltl_heuristic_layout.end(), [](const char c) { return c != '0' && c != '1'; })) {
            std::cerr << "--ltl-heuristic-config should only contain 0 or 1." << std::endl;
            return 1;
        }
        if (options.ltl_heuristic_layout[0] != '1') {
            std::cerr << "--ltl-heuristic-config should start with 1 to enable LTL heuristic for the anchor level." << std::endl;
            return 1;
        }
        std::memcpy(level_ltl_chars, options.ltl_heuristic_layout.c_str(), options.ltl_heuristic_layout.length());

        if (options.llm_heuristic_layout.empty()) {
            std::cerr << "--llm-heuristic-config is not provided." << std::endl;
            return 1;
        }
        if (options.llm_heuristic_layout.length() != static_cast<std::size_t>(options.max_level) + 2) {
            std::cerr << "--llm-heuristic-config is not valid with max_level = " << max_level_str << std::endl;
            return 1;
        }
        if (std::any_of(options.llm_heuristic_layout.begin(), options.llm_heuristic_layout.end(), [](const char c) { return c != '0' && c != '1'; })) {
            std::cerr << "--llm-heuristic-config should only contain 0 or 1." << std::endl;
            return 1;
        }
        if (options.llm_heuristic_layout[0] != '0') {
            std::cerr << "--llm-heuristic-layout should start with 0 because anchor level does not allow LLM heuristic." << std::endl;
            return 1;
        }
        std::memcpy(level_llm_chars, options.llm_heuristic_layout.c_str(), options.llm_heuristic_layout.length());

        use_llm_heuristic = std::any_of(options.ltl_heuristic_layout.begin(), options.ltl_heuristic_layout.end(), [](const char c) { return c == '1'; });
        if (use_llm_heuristic && options.llm_heuristic_file.empty()) {
            std::cerr << "--llm-heuristic-file is not provided." << std::endl;
            return 1;
        }
        if (options.repeat < 1) {
            std::cerr << "repeat should be at least 1." << std::endl;
            return 1;
        }
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (options.map_data_dir.empty()) {
        options.map_data_dir = std::filesystem::path(options.scene_graph_file).parent_path().string();
        ERL_INFO("--map-data-dir is not provided, using the default value: %s", options.map_data_dir.c_str());
    }

    // create output dir
    if (!std::filesystem::exists(options.output_dir)) { std::filesystem::create_directories(options.output_dir); }
    // load scene graph
    auto scene_graph = std::make_shared<scene_graph::Building>();
    scene_graph->FromYamlFile(options.scene_graph_file);
    // load the env setting
    auto env_setting = std::make_shared<EnvironmentLTLSceneGraph::Setting>();
    env_setting->data_dir = options.map_data_dir;
    using AutFileType = FiniteStateAutomaton::Setting::FileType;
    env_setting->fsa = std::make_shared<FiniteStateAutomaton::Setting>(options.automaton_file, AutFileType::kSpotHoa, options.make_automaton_complete);
    env_setting->LoadAtomicPropositions(options.ap_file);
    // create the environment
    env_setting->max_level = options.max_level;
    if (options.robot_radius > 0) {
        long n = 360;
        Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(n + 1, 0, 2 * M_PI).head(n);
        env_setting->shape.resize(2, n);
        for (long i = 0; i < n; ++i) {
            env_setting->shape(0, i) = options.robot_radius * cos(angles[i]);
            env_setting->shape(1, i) = options.robot_radius * sin(angles[i]);
        }
    }
    env_setting->object_reach_distance = options.object_reach_radius;
    auto env = std::make_shared<EnvironmentLTLSceneGraph>(scene_graph, env_setting);
    // get initial states, goal set and goal tolerance.
    auto init_q = static_cast<int>(env_setting->fsa->initial_state);
    Eigen::VectorXd start = env->GridToMetric(Eigen::Vector4i(options.init_grid_x, options.init_grid_y, options.init_grid_z, init_q));
    auto num_goals = static_cast<long>(env_setting->fsa->accepting_states.size());
    ERL_ASSERTM(num_goals > 0, "no accepting states in the automaton.");
    std::vector<Eigen::VectorXd> goals(num_goals);
    for (long i = 0; i < num_goals; ++i) {
        Eigen::VectorXd &goal = goals[i];
        goal.resize(4);
        goal[0] = 0;  // does not matter
        goal[1] = 0;  // does not matter
        goal[2] = 0;  // does not matter
        goal[3] = static_cast<double>(env_setting->fsa->accepting_states[i]);
    }
    double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd goal_tolerance = Eigen::Vector4d(inf, inf, inf, 0);
    // get ltl heuristic and llm heuristic
    ERL_INFO(
        hierarchy_layout,
        level_enabled_chars[0],
        level_enabled_chars[1],
        level_enabled_chars[2],
        level_enabled_chars[3],
        level_enabled_chars[4],
        level_ltl_chars[0],
        level_ltl_chars[1],
        level_ltl_chars[2],
        level_ltl_chars[3],
        level_ltl_chars[4],
        level_llm_chars[0],
        level_llm_chars[1],
        level_llm_chars[2],
        level_llm_chars[3],
        level_llm_chars[4]);
    auto ltl_heuristic = std::make_shared<LinearTemporalLogicHeuristic3D>(env->GetFiniteStateAutomaton(), env->GetLabelMaps(), env->GetGridMapInfo());
    std::shared_ptr<LLMSceneGraphHeuristic> llm_heuristic = nullptr;
    if (use_llm_heuristic) {
        auto llm_heuristic_setting = std::make_shared<LLMSceneGraphHeuristic::Setting>();
        llm_heuristic_setting->FromYamlFile(options.llm_heuristic_file);
        llm_heuristic = std::make_shared<LLMSceneGraphHeuristic>(llm_heuristic_setting, env);
    }
    std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics;
    for (std::size_t i = 0; i < options.ltl_heuristic_layout.length(); ++i) {
        if (options.ltl_heuristic_layout[i] == '1') { heuristics.emplace_back(ltl_heuristic, i); }
    }
    for (std::size_t i = 0; i < options.llm_heuristic_layout.length(); ++i) {
        if (options.llm_heuristic_layout[i] == '1') { heuristics.emplace_back(llm_heuristic, i); }
    }
    // create the planner
    auto planning_interface = std::make_shared<PlanningInterfaceMultiResolutions>(env, heuristics, start, goals, std::vector{goal_tolerance});
    auto amra_setting = std::make_shared<amra_star::AMRAStar::Setting>();
    amra_setting->log = false;
    std::shared_ptr<amra_star::Output> result;
    for (int i = 0; i < options.repeat; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        amra_star::AMRAStar planner(planning_interface, amra_setting);
        auto t1 = std::chrono::high_resolution_clock::now();
        result = planner.Plan();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "[" << i << "]Planner construction time: " << std::chrono::duration<double, std::micro>(t1 - t0).count() << " us." << std::endl;
        std::cout << "[" << i << "]Planning time: " << std::chrono::duration<double, std::micro>(t2 - t1).count() << " us." << std::endl;
    }
    // save the experiment setting
    std::filesystem::path output_dir = options.output_dir;
    options.AsYamlFile(output_dir / "experiment_setting.yaml");
    // save the result
    if (options.save_amra_log) {
        std::cout << "Running AMRA* again with logging enabled." << std::endl;
        amra_setting->log = true;
        amra_star::AMRAStar planner(planning_interface, amra_setting);
        result = planner.Plan();
        std::cout << "Saving AMRA* log..." << std::endl;
        result->Save(output_dir / "amra.solution");
    }
    // draw path
    for (const auto &[plan_itr, amra_path]: result->paths) {
        long num_points = amra_path.cols();
        std::vector<cv::Point2i> cv_path;  // floor -> path
        int img_idx = 0;
        int cur_floor = -1;
        for (long i = 0; i < num_points; ++i) {
            Eigen::Vector4d p = amra_path.col(i);
            Eigen::Vector4i grid = env->MetricToGrid(p);
            if (cur_floor < 0) { cur_floor = grid[2]; }
            if (cur_floor != grid[2]) {
                cv::Mat img;
                img = erl::common::ColorGrayCustom(scene_graph->LoadCatMap(options.map_data_dir, cur_floor));
                cv::polylines(img, cv_path, false, cv::Scalar(0, 0, 0), 2);
                // draw start and goal
                cv::circle(img, cv_path.front(), 10, cv::Scalar(0, 0, 255), -1);
                cv::circle(img, cv_path.back(), 10, cv::Scalar(0, 255, 0), -1);
                std::string img_name = erl::common::AsString("plan_", plan_itr, "_img_", img_idx, "_floor_", cur_floor);
                if (options.hold_for_visualization) { cv::imshow(img_name, img); }
                cv::imwrite(output_dir / (img_name + ".png"), img);
                img_idx++;
                cur_floor = grid[2];
                cv_path.clear();
            }
            cv_path.emplace_back(grid[1], grid[0]);
        }
        cv::Mat img;
        img = erl::common::ColorGrayCustom(scene_graph->LoadCatMap(options.map_data_dir, cur_floor));
        cv::polylines(img, cv_path, false, cv::Scalar(0, 0, 0), 2);
        // draw start and goal
        cv::circle(img, cv_path.front(), 10, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, cv_path.back(), 10, cv::Scalar(0, 255, 0), -1);
        std::string img_name = erl::common::AsString("plan_", plan_itr, "_img_", img_idx, "_floor_", cur_floor);
        if (options.hold_for_visualization) { cv::imshow(img_name, img); }
        cv::imwrite(output_dir / (img_name + ".png"), img);
    }
    // hold for visualization
    if (options.hold_for_visualization) { cv::waitKey(0); }

    return 0;
}

namespace YAML {
    template<>
    struct convert<Options> {
        static Node
        encode(const Options &rhs) {
            Node node;
            node["output_dir"] = rhs.output_dir;
            node["scene_graph_file"] = rhs.scene_graph_file;
            node["map_data_dir"] = rhs.map_data_dir;
            node["automaton_file"] = rhs.automaton_file;
            node["ap_file"] = rhs.ap_file;
            node["llm_heuristic_file"] = rhs.llm_heuristic_file;
            node["init_grid_x"] = rhs.init_grid_x;
            node["init_grid_y"] = rhs.init_grid_y;
            node["init_grid_z"] = rhs.init_grid_z;
            node["robot_radius"] = rhs.robot_radius;
            node["object_reach_radius"] = rhs.object_reach_radius;
            node["max_level"] = rhs.max_level;
            node["ltl_heuristic_layout"] = rhs.ltl_heuristic_layout;
            node["llm_heuristic_layout"] = rhs.llm_heuristic_layout;
            node["repeat"] = rhs.repeat;
            node["save_amra_log"] = rhs.save_amra_log;
            node["hold_for_visualization"] = rhs.hold_for_visualization;
            return node;
        }

        static bool
        decode(const Node &node, Options &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.output_dir = node["output_dir"].as<std::string>();
            rhs.scene_graph_file = node["scene_graph_file"].as<std::string>();
            rhs.map_data_dir = node["map_data_dir"].as<std::string>();
            rhs.automaton_file = node["automaton_file"].as<std::string>();
            rhs.ap_file = node["ap_file"].as<std::string>();
            rhs.llm_heuristic_file = node["llm_heuristic_file"].as<std::string>();
            rhs.init_grid_x = node["init_grid_x"].as<int>();
            rhs.init_grid_y = node["init_grid_y"].as<int>();
            rhs.init_grid_z = node["init_grid_z"].as<int>();
            rhs.robot_radius = node["robot_radius"].as<double>();
            rhs.object_reach_radius = node["object_reach_radius"].as<double>();
            rhs.max_level = node["max_level"].as<erl::env::scene_graph::Node::Type>();
            rhs.ltl_heuristic_layout = node["ltl_heuristic_layout"].as<std::string>();
            rhs.llm_heuristic_layout = node["llm_heuristic_layout"].as<std::string>();
            rhs.repeat = node["repeat"].as<int>();
            rhs.save_amra_log = node["save_amra_log"].as<bool>();
            rhs.hold_for_visualization = node["hold_for_visualization"].as<bool>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const Options &rhs) {
        out << BeginMap;
        out << Key << "output_dir" << Value << rhs.output_dir;
        out << Key << "scene_graph_file" << Value << rhs.scene_graph_file;
        out << Key << "map_data_dir" << Value << rhs.map_data_dir;
        out << Key << "automaton_file" << Value << rhs.automaton_file;
        out << Key << "ap_file" << Value << rhs.ap_file;
        out << Key << "llm_heuristic_file" << Value << rhs.llm_heuristic_file;
        out << Key << "init_grid_x" << Value << rhs.init_grid_x;
        out << Key << "init_grid_y" << Value << rhs.init_grid_y;
        out << Key << "init_grid_z" << Value << rhs.init_grid_z;
        out << Key << "robot_radius" << Value << rhs.robot_radius;
        out << Key << "object_reach_radius" << Value << rhs.object_reach_radius;
        out << Key << "max_level" << Value << rhs.max_level;
        out << Key << "ltl_heuristic_layout" << Value << rhs.ltl_heuristic_layout;
        out << Key << "llm_heuristic_layout" << Value << rhs.llm_heuristic_layout;
        out << Key << "repeat" << Value << rhs.repeat;
        out << Key << "save_amra_log" << Value << rhs.save_amra_log;
        out << Key << "hold_for_visualization" << Value << rhs.hold_for_visualization;
        out << EndMap;
        return out;
    }
}  // namespace YAML
