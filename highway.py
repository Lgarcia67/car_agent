import gymnasium as gym
import highway_env


class Highway:
    def __init__(self):
        self.config = self.setConfig()
        self.env = gym.make("highway-fast-v0", config=self.config, render_mode="human")

    def setConfig(self):
        return {
            "lanes_count": 3,
            "policy_frequency": 2,
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
                },
            "collision_reward": -5,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": -0.75,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
            
        }
    def reset(self, seed=42):
        return self.env.reset(seed=seed)

