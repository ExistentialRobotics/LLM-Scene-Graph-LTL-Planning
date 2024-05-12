#include <filesystem>
#include "erl_common/test_helper.hpp"
#include "erl_env/environment_scene_graph.hpp"
#include "erl_env/environment_ltl_scene_graph.hpp"
#include "erl_search_planning/amra_star.hpp"
#include "erl_search_planning/heuristic.hpp"
#include "erl_search_planning/ltl_3d_heuristic.hpp"
#include "erl_search_planning/llm_scene_graph_heuristic.hpp"

TEST(AMRAStarSceneGraph, SingleFloor) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::env;
    using namespace erl::search_planning;

    // load environment
    std::filesystem::path path = gtest_src_dir / "building.yaml";
    std::filesystem::path output_dir = gtest_src_dir / "results" / test_output_dir;
    std::filesystem::create_directories(output_dir);
    auto building = std::make_shared<scene_graph::Building>();
    building->FromYamlFile(path.string());

    auto env_setting = std::make_shared<EnvironmentSceneGraph::Setting>();
    env_setting->data_dir = gtest_src_dir.string();
    env_setting->shape = Eigen::Matrix2Xd(2, 360);
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(360, 0, 2 * M_PI);
    double r = 0.1;
    for (int i = 0; i < 360; ++i) {
        env_setting->shape(0, i) = r * cos(angles[i]);
        env_setting->shape(1, i) = r * sin(angles[i]);
    }
    auto env_scene_graph = std::make_shared<EnvironmentSceneGraph>(building, env_setting);

    Eigen::VectorXd start = env_scene_graph->GridToMetric(Eigen::Vector3i(400, 900, 1));
    Eigen::VectorXd goal = env_scene_graph->GridToMetric(Eigen::Vector3i(450, 200, 1));
    Eigen::VectorXd goal_tolerance = Eigen::VectorXd::Zero(3);
    std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics = {
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 0},  // anchor
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 1},  // kNA
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 2},  // kObject
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 3},  // kRoom
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 4},  // kFloor
    };

    auto planning_interface = std::make_shared<PlanningInterfaceMultiResolutions>(
        env_scene_graph,
        heuristics,
        start,
        std::vector<Eigen::VectorXd>{goal},
        std::vector<Eigen::VectorXd>{goal_tolerance});
    auto amra_setting = std::make_shared<amra_star::AMRAStar::Setting>();
    amra_setting->log = true;
    amra_star::AMRAStar planner(planning_interface, amra_setting);

    std::shared_ptr<amra_star::Output> result;
    erl::common::ReportTime<std::chrono::microseconds>(test_info->name(), 0, true, [&]() { result = planner.Plan(); });
    double path_cost = result->costs[result->latest_plan_itr];
    std::cout << "Path cost: " << path_cost << std::endl;
    EXPECT_NEAR(path_cost, 8.701223, 1e-6);
    if (amra_setting->log) { result->Save(output_dir / "amra.solution"); }

    // draw path
    for (auto &itr: result->paths) {
        uint32_t plan_itr = itr.first;
        Eigen::Matrix3Xd amra_path = itr.second;
        long num_points = amra_path.cols();
        cv::Mat cat_map = erl::common::ColorGrayCustom(building->LoadCatMap(gtest_src_dir, 1));
        std::vector<cv::Point2i> cv_path;
        cv_path.reserve(num_points);
        for (long i = 0; i < num_points; ++i) {
            Eigen::Vector3d p = amra_path.col(i);
            Eigen::Vector3i grid = env_scene_graph->MetricToGrid(p);
            cv_path.emplace_back(grid[1], grid[0]);
        }
        cv::polylines(cat_map, cv_path, false, cv::Scalar(0, 0, 255), 2);
        cv::imshow("plan_" + std::to_string(plan_itr), cat_map);
        cv::imwrite(output_dir / erl::common::AsString("plan_", plan_itr, ".png"), cat_map);
    }
    cv::waitKey(1000);
}

TEST(AMRAStarSceneGraph, CrossFloor) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::env;
    using namespace erl::search_planning;

    // load environment
    std::filesystem::path building_path = gtest_src_dir / "building.yaml";
    std::filesystem::path output_dir = gtest_src_dir / "results" / test_output_dir;
    std::filesystem::create_directories(output_dir);
    auto building = std::make_shared<scene_graph::Building>();
    building->FromYamlFile(building_path.string());

    auto env_setting = std::make_shared<EnvironmentSceneGraph::Setting>();
    env_setting->data_dir = gtest_src_dir.string();
    env_setting->shape = Eigen::Matrix2Xd(2, 360);
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(360, 0, 2 * M_PI);
    double r = 0.1;
    for (int i = 0; i < 360; ++i) {
        env_setting->shape(0, i) = r * cos(angles[i]);
        env_setting->shape(1, i) = r * sin(angles[i]);
    }
    auto env_scene_graph = std::make_shared<EnvironmentSceneGraph>(building, env_setting);

    Eigen::VectorXd start = env_scene_graph->GridToMetric(Eigen::Vector3i(400, 900, 1));
    Eigen::VectorXd goal = env_scene_graph->GridToMetric(Eigen::Vector3i(400, 800, 2));
    Eigen::VectorXd goal_tolerance = Eigen::VectorXd::Zero(3);
    std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics = {
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 0},  // anchor
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 1},  // kNA
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 2},  // kObject
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 3},  // kRoom
        {std::make_shared<EuclideanDistanceHeuristic<3>>(goal, goal_tolerance), 4},  // kFloor
    };

    auto planning_interface = std::make_shared<PlanningInterfaceMultiResolutions>(
        env_scene_graph,
        heuristics,
        start,
        std::vector<Eigen::VectorXd>{goal},
        std::vector<Eigen::VectorXd>{goal_tolerance});
    auto amra_setting = std::make_shared<amra_star::AMRAStar::Setting>();
    amra_setting->log = true;
    amra_star::AMRAStar planner(planning_interface, amra_setting);

    std::shared_ptr<amra_star::Output> result;
    erl::common::ReportTime<std::chrono::microseconds>(test_info->name(), 0, true, [&]() { result = planner.Plan(); });
    double path_cost = result->costs[result->latest_plan_itr];
    std::cout << "Path cost: " << path_cost << std::endl;
    EXPECT_NEAR(path_cost, 15.485639, 1e-6);
    if (amra_setting->log) { result->Save(output_dir / "amra.solution"); }

    // draw path
    for (auto &itr: result->paths) {
        uint32_t plan_itr = itr.first;
        Eigen::Matrix3Xd amra_path = itr.second;
        long num_points = amra_path.cols();
        std::unordered_map<int, std::vector<cv::Point2i>> cv_paths;  // floor -> path
        cv_paths.reserve(num_points);
        for (long i = 0; i < num_points; ++i) {
            Eigen::Vector3d p = amra_path.col(i);
            Eigen::Vector3i grid = env_scene_graph->MetricToGrid(p);
            cv_paths[grid[2]].emplace_back(grid[1], grid[0]);
        }
        std::unordered_map<int, cv::Mat> cat_maps;  // floor -> map
        for (auto &[floor_num, cv_path]: cv_paths) {
            auto &cat_map = cat_maps[floor_num];
            cat_map = erl::common::ColorGrayCustom(building->LoadCatMap(gtest_src_dir, floor_num));
            cv::polylines(cat_map, cv_path, false, cv::Scalar(0, 0, 255), 2);
            cv::imshow(erl::common::AsString("plan_", plan_itr, "_floor_", floor_num), cat_map);
            cv::imwrite(output_dir / erl::common::AsString("plan_", plan_itr, "_floor_", floor_num, ".png"), cat_map);
        }
    }
    cv::waitKey(1000);  // wait 1s
}

TEST(AMRAStarSceneGraph, LinearTemporalLogic) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::env;
    using namespace erl::search_planning;

    std::filesystem::path building_path = gtest_src_dir / "building.yaml";
    std::filesystem::path output_dir = gtest_src_dir / "results" / test_output_dir;
    std::filesystem::create_directories(output_dir);

    // load the building
    auto building = std::make_shared<scene_graph::Building>();
    building->FromYamlFile(building_path.string());

    // load the env setting
    auto env_setting = std::make_shared<EnvironmentLTLSceneGraph::Setting>();
    env_setting->data_dir = gtest_src_dir.string();
    env_setting->shape = Eigen::Matrix2Xd(2, 360);
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(360, 0, 2 * M_PI);
    double r = 0.1;
    for (int i = 0; i < 360; ++i) {
        env_setting->shape(0, i) = r * cos(angles[i]);
        env_setting->shape(1, i) = r * sin(angles[i]);
    }
    // load the finite state automaton setting from spot hoa file
    env_setting->fsa = std::make_shared<FiniteStateAutomaton::Setting>(gtest_src_dir / "automaton.aut", FiniteStateAutomaton::Setting::FileType::kSpotHoa);
    // load the atomic propositions
    env_setting->LoadAtomicPropositions(gtest_src_dir / "ap_desc.yaml");

    // create the environment
    auto environment_ltl_scene_graph = std::make_shared<EnvironmentLTLSceneGraph>(building, env_setting);

    Eigen::VectorXd start = environment_ltl_scene_graph->GridToMetric(Eigen::Vector4i(100, 200, 0, int(env_setting->fsa->initial_state)));
    Eigen::VectorXd goal = environment_ltl_scene_graph->GridToMetric(Eigen::Vector4i(0, 0, 0, int(env_setting->fsa->accepting_states[0])));
    double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd goal_tolerance = Eigen::Vector4d(inf, inf, inf, 0);
    auto ltl_heuristic = std::make_shared<LinearTemporalLogicHeuristic3D>(
        environment_ltl_scene_graph->GetFiniteStateAutomaton(),
        environment_ltl_scene_graph->GetLabelMaps(),
        environment_ltl_scene_graph->GetGridMapInfo());
    std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics = {
        {ltl_heuristic, 0},  // anchor
        {ltl_heuristic, 1},  // kOCC
        {ltl_heuristic, 2},  // kObject
        {ltl_heuristic, 3},  // kRoom
        {ltl_heuristic, 4},  // kFloor
    };

    auto planning_interface = std::make_shared<PlanningInterfaceMultiResolutions>(
        environment_ltl_scene_graph,
        heuristics,
        start,
        std::vector<Eigen::VectorXd>{goal},
        std::vector<Eigen::VectorXd>{goal_tolerance});
    auto amra_setting = std::make_shared<amra_star::AMRAStar::Setting>();
    amra_setting->log = true;
    amra_star::AMRAStar planner(planning_interface, amra_setting);

    std::shared_ptr<amra_star::Output> result;
    erl::common::ReportTime<std::chrono::microseconds>(test_info->name(), 0, true, [&]() { result = planner.Plan(); });
    double path_cost = result->costs[result->latest_plan_itr];
    std::cout << "Path cost: " << path_cost << std::endl;
    EXPECT_NEAR(path_cost, 18.562500852412878, 1e-6);
    if (amra_setting->log) { result->Save(output_dir / "amra.solution"); }

    // draw path
    for (auto &itr: result->paths) {
        uint32_t plan_itr = itr.first;
        Eigen::Matrix4Xd amra_path = itr.second;
        long num_points = amra_path.cols();
        std::unordered_map<int, std::vector<cv::Point2i>> cv_paths;  // floor -> path
        cv_paths.reserve(num_points);
        for (long i = 0; i < num_points; ++i) {
            Eigen::Vector4d p = amra_path.col(i);
            Eigen::Vector4i grid = environment_ltl_scene_graph->MetricToGrid(p);
            cv_paths[grid[2]].emplace_back(grid[1], grid[0]);
        }
        std::unordered_map<int, cv::Mat> cat_maps;  // floor -> map
        for (auto &[floor_num, cv_path]: cv_paths) {
            auto &cat_map = cat_maps[floor_num];
            cat_map = erl::common::ColorGrayCustom(building->LoadCatMap(gtest_src_dir, floor_num));
            cv::polylines(cat_map, cv_path, false, cv::Scalar(0, 0, 255), 2);
            cv::imshow(erl::common::AsString("plan_", plan_itr, "_floor_", floor_num), cat_map);
            cv::imwrite(output_dir / erl::common::AsString("plan_", plan_itr, "_floor_", floor_num, ".png"), cat_map);
        }
    }
    cv::waitKey(1000);  // wait 1s
}

TEST(AMRAStarSceneGraph, SingleLayer) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::env;
    using namespace erl::search_planning;

    std::filesystem::path building_path =  gtest_src_dir / "building.yaml";
    std::filesystem::path output_dir = gtest_src_dir / "results" / test_output_dir;
    std::filesystem::create_directories(output_dir);

    // load the building
    auto building = std::make_shared<scene_graph::Building>();
    building->FromYamlFile(building_path.string());

    // load the env setting
    auto env_setting = std::make_shared<EnvironmentLTLSceneGraph::Setting>();
    env_setting->data_dir = gtest_src_dir.string();
    env_setting->shape = Eigen::Matrix2Xd(2, 360);
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(360, 0, 2 * M_PI);
    double r = 0.1;
    for (int i = 0; i < 360; ++i) {
        env_setting->shape(0, i) = r * cos(angles[i]);
        env_setting->shape(1, i) = r * sin(angles[i]);
    }
    // load the finite state automaton setting from spot hoa file
    env_setting->fsa = std::make_shared<FiniteStateAutomaton::Setting>(gtest_src_dir / "automaton.aut", FiniteStateAutomaton::Setting::FileType::kSpotHoa);
    // load the atomic propositions
    env_setting->LoadAtomicPropositions(gtest_src_dir / "ap_desc.yaml");

    // create the environment
    env_setting->max_level = erl::env::scene_graph::Node::Type::kOcc;
    auto environment_ltl_scene_graph = std::make_shared<EnvironmentLTLSceneGraph>(building, env_setting);

    Eigen::VectorXd start = environment_ltl_scene_graph->GridToMetric(Eigen::Vector4i(100, 200, 0, int(env_setting->fsa->initial_state)));
    Eigen::VectorXd goal = environment_ltl_scene_graph->GridToMetric(Eigen::Vector4i(0, 0, 0, int(env_setting->fsa->accepting_states[0])));
    double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd goal_tolerance = Eigen::Vector4d(inf, inf, inf, 0);
    auto ltl_heuristic = std::make_shared<LinearTemporalLogicHeuristic3D>(
        environment_ltl_scene_graph->GetFiniteStateAutomaton(),
        environment_ltl_scene_graph->GetLabelMaps(),
        environment_ltl_scene_graph->GetGridMapInfo());

    std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics = {
        {ltl_heuristic, 0},  // anchor
        {ltl_heuristic, 1},  // kNA
    };

    auto planning_interface = std::make_shared<PlanningInterfaceMultiResolutions>(
        environment_ltl_scene_graph,
        heuristics,
        start,
        std::vector<Eigen::VectorXd>{goal},
        std::vector<Eigen::VectorXd>{goal_tolerance});
    auto amra_setting = std::make_shared<amra_star::AMRAStar::Setting>();
    amra_setting->log = true;
    amra_star::AMRAStar planner(planning_interface, amra_setting);

    std::shared_ptr<amra_star::Output> result;
    erl::common::ReportTime<std::chrono::microseconds>(test_info->name(), 0, true, [&]() { result = planner.Plan(); });
    double path_cost = result->costs[result->latest_plan_itr];
    std::cout << "Path cost: " << path_cost << std::endl;
    EXPECT_NEAR(path_cost, 18.562500852413102, 1e-6);
    if (amra_setting->log) { result->Save(output_dir / "amra.solution"); }

    // draw path
    for (auto &itr: result->paths) {
        uint32_t plan_itr = itr.first;
        Eigen::Matrix4Xd amra_path = itr.second;
        long num_points = amra_path.cols();
        std::unordered_map<int, std::vector<cv::Point2i>> cv_paths;  // floor -> path
        cv_paths.reserve(num_points);
        for (long i = 0; i < num_points; ++i) {
            Eigen::Vector4d p = amra_path.col(i);
            Eigen::Vector4i grid = environment_ltl_scene_graph->MetricToGrid(p);
            cv_paths[grid[2]].emplace_back(grid[1], grid[0]);
        }
        std::unordered_map<int, cv::Mat> cat_maps;  // floor -> map
        for (auto &[floor_num, cv_path]: cv_paths) {
            auto &cat_map = cat_maps[floor_num];
            cat_map = erl::common::ColorGrayCustom(building->LoadCatMap(gtest_src_dir, floor_num));
            cv::polylines(cat_map, cv_path, false, cv::Scalar(0, 0, 255), 2);
            cv::imshow(erl::common::AsString("plan_", plan_itr, "_floor_", floor_num), cat_map);
            cv::imwrite(output_dir / erl::common::AsString("plan_", plan_itr, "_floor_", floor_num, ".png"), cat_map);
        }
    }
    cv::waitKey(1000);  // wait 1s
}

TEST(LLMSceneGraph, Heuristic) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::env;
    using namespace erl::search_planning;

    std::filesystem::path building_path = gtest_src_dir / "building.yaml";
    std::filesystem::path output_dir = gtest_src_dir / "results" / test_output_dir;
    std::filesystem::create_directories(output_dir);

    // load the building
    auto building = std::make_shared<scene_graph::Building>();
    building->FromYamlFile(building_path.string());

    // load the env setting
    auto env_setting = std::make_shared<EnvironmentLTLSceneGraph::Setting>();
    env_setting->data_dir = gtest_src_dir.string();
    env_setting->object_reach_distance = 0.6;
    env_setting->shape = Eigen::Matrix2Xd(2, 360);
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(360, 0, 2 * M_PI);
    double r = 0.1;
    for (int i = 0; i < 360; ++i) {
        env_setting->shape(0, i) = r * cos(angles[i]);
        env_setting->shape(1, i) = r * sin(angles[i]);
    }
    // load the finite state automaton setting from spot hoa file
    env_setting->fsa = std::make_shared<FiniteStateAutomaton::Setting>(gtest_src_dir / "automaton.aut", FiniteStateAutomaton::Setting::FileType::kSpotHoa);
    // load the atomic propositions
    env_setting->LoadAtomicPropositions(gtest_src_dir / "ap_desc.yaml");

    // create the environment
    auto environment_ltl_scene_graph = std::make_shared<EnvironmentLTLSceneGraph>(building, env_setting);

    Eigen::VectorXd start = environment_ltl_scene_graph->GridToMetric(Eigen::Vector4i(100, 200, 0, int(env_setting->fsa->initial_state)));
    Eigen::VectorXd goal = environment_ltl_scene_graph->GridToMetric(Eigen::Vector4i(0, 0, 0, int(env_setting->fsa->accepting_states[0])));
    double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd goal_tolerance = Eigen::Vector4d(inf, inf, inf, 0);
    auto ltl_heuristic = std::make_shared<LinearTemporalLogicHeuristic3D>(
        environment_ltl_scene_graph->GetFiniteStateAutomaton(),
        environment_ltl_scene_graph->GetLabelMaps(),
        environment_ltl_scene_graph->GetGridMapInfo());

    auto gpt4_heuristic_file = gtest_src_dir / "gpt4_path_v2.yaml";
    auto gpt4_heuristic_setting = std::make_shared<LLMSceneGraphHeuristic::Setting>();
    gpt4_heuristic_setting->FromYamlFile(gpt4_heuristic_file.string());
    auto gpt4_heuristic = std::make_shared<LLMSceneGraphHeuristic>(gpt4_heuristic_setting, environment_ltl_scene_graph);

    std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics = {
        {ltl_heuristic, 0},  // anchor
        {ltl_heuristic, 1},  // kNA
        {gpt4_heuristic, 1},
        {ltl_heuristic, 2},  // kObject
        {gpt4_heuristic, 2},
        {ltl_heuristic, 3},  // kRoom
        {gpt4_heuristic, 3},
        {ltl_heuristic, 4},  // kFloor
        {gpt4_heuristic, 4},
    };

    auto planning_interface = std::make_shared<PlanningInterfaceMultiResolutions>(
        environment_ltl_scene_graph,
        heuristics,
        start,
        std::vector<Eigen::VectorXd>{goal},
        std::vector<Eigen::VectorXd>{goal_tolerance});
    auto amra_setting = std::make_shared<amra_star::AMRAStar::Setting>();
    amra_setting->log = true;
    amra_star::AMRAStar planner(planning_interface, amra_setting);

    std::shared_ptr<amra_star::Output> result;
    erl::common::ReportTime<std::chrono::microseconds>(test_info->name(), 0, true, [&]() { result = planner.Plan(); });
    double path_cost = result->costs[result->latest_plan_itr];
    std::cout << "Path cost: " << path_cost << std::endl;
    EXPECT_NEAR(path_cost, 18.562500852412878, 1e-6);
    if (amra_setting->log) { result->Save(output_dir / "amra.solution"); }

    // draw path
    for (auto &itr: result->paths) {
        uint32_t plan_itr = itr.first;
        Eigen::Matrix4Xd amra_path = itr.second;
        long num_points = amra_path.cols();
        std::unordered_map<int, std::vector<cv::Point2i>> cv_paths;  // floor -> path
        cv_paths.reserve(num_points);
        for (long i = 0; i < num_points; ++i) {
            Eigen::Vector4d p = amra_path.col(i);
            Eigen::Vector4i grid = environment_ltl_scene_graph->MetricToGrid(p);
            cv_paths[grid[2]].emplace_back(grid[1], grid[0]);
        }
        std::unordered_map<int, cv::Mat> cat_maps;  // floor -> map
        for (auto &[floor_num, cv_path]: cv_paths) {
            auto &cat_map = cat_maps[floor_num];
            cat_map = erl::common::ColorGrayCustom(building->LoadCatMap(gtest_src_dir, floor_num));
            cv::polylines(cat_map, cv_path, false, cv::Scalar(0, 0, 255), 2);
            cv::imshow(erl::common::AsString("plan_", plan_itr, "_floor_", floor_num), cat_map);
            cv::imwrite(output_dir / erl::common::AsString("plan_", plan_itr, "_floor_", floor_num, ".png"), cat_map);
        }
    }
    cv::waitKey(1000);  // wait 1s
}
