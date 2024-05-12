from setuptools import setup
from setuptools import find_packages

setup(
    name="llm-planning",
    version="2.0",
    description="Large scale path planning with scene graph and large language model",
    author="Arash Asgharivaskasi, Thai Duong, Zhirui Dai",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    include_package_data=True,
    package_data={"": ["*.txt", "*.yaml", "*.aut", "*.npz"]},
    entry_points=dict(
        console_scripts=[
            "erl-gibson-scene-graph=llm_planning.scene_graph.gibson_scene_graph:main",
            "erl-nl2ltl=llm_planning.natural_to_temporal_logic.translator:main",
            "erl-llm-heuristic=llm_planning.llm_heuristic.gpt_heuristics_taskgen_auto:main",
        ]
    ),
)
