from dataclasses import dataclass
import tyro
import numpy as np
from gr00t import server_client


@dataclass
class ClientConfig:
    host: str = "172.16.78.10"
    port: int = 34857
    timeout_ms: int = 15000
    api_token: str | None = None


if __name__ == "__main__":
    config = tyro.cli(ClientConfig)
    client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
        api_token=config.api_token,
    )

    if client.ping():
        print("Server is alive!")
    else:
        print("Failed to connect to the server.")

    ### Example of calling an endpoint of get action
    observation = dict()
    video = {
        "ego_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
    }

    language = {
        "annotation.human.task_description": [[""]]
    }

    state = {
        "base_translation": np.zeros((1, 1, 3), dtype=np.float32),
        "base_rotation": np.zeros((1, 1, 4), dtype=np.float32),
        "left_leg": np.zeros((1, 1, 6), dtype=np.float32),
        "right_leg": np.zeros((1, 1, 6), dtype=np.float32),
        "waist": np.zeros((1, 1, 3), dtype=np.float32),
        "left_arm": np.zeros((1, 1, 7), dtype=np.float32),
        "right_arm": np.zeros((1, 1, 7), dtype=np.float32),
    }

    stickman = {
        "annotation.human.stickman": np.zeros((1, 1, 900), dtype=np.float32)
    }
    observation.update({"video": video, "language": language, "state": state, "stickman": stickman})
    client.get_action(observation)
