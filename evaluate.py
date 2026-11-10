import torch

from agent import Agent
from highway import Highway


class Evaluate:
    def __init__(self, agent):
        self.agent = agent

    def evaluate(self, seed=42):
        highway = Highway()
        env = highway.env
        total_reward = 0.0

        try:
            obs, _ = highway.reset(seed=seed)

            while True:
                state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0

                with torch.no_grad():
                    action = self.agent(state).argmax(dim=1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    return total_reward
        finally:
            env.close()


if __name__ == "__main__":
    agent = Agent()
    evaluator = Evaluate(agent)
    fitness = evaluator.evaluate()
    print("Agent fitness:", fitness)



    