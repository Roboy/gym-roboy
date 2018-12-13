import os

REQUIRED_VARIABLES = [
    "ROS2_WS",
    "ROS2_ROBOY_WS",
    "CARDSFLOW_WS",
]


def test_environment_variables_are_set():
    for var in REQUIRED_VARIABLES:
        assert os.environ[var] is not None
