import random
from collections import deque

import numpy as np
import torch
from torch import nn

from agent import Agent
from highway import Highway


def train(num_episodes=1000, learning_starts=256, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    highway = Highway()
    env = highway.env

    try:
        env.action_space.seed(seed)
        num_actions = env.action_space.n

        online = Agent(num_actions=num_actions).to(device)
        target = Agent(num_actions=num_actions).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()

        optimizer = torch.optim.Adam(online.parameters(), lr=0.0003)
        loss_fn = nn.SmoothL1Loss()
        replay = deque(maxlen=3000)

        batch_size = 16
        gamma = 0.99
        steps = 0
        updates = 0

        for episode in range(1, num_episodes + 1):
            obs, _ = highway.reset(seed=seed if episode == 1 else None)
            episode_reward = 0.0

            while True:
                epsilon = max(0.05, 1.0 - 0.95 * steps / 10_000)

                if steps < learning_starts or random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0) / 255.0

                    with torch.no_grad():
                        action = online(state).argmax(dim=1).item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                replay.append(
                    (obs.copy(), action, reward, next_obs.copy(), terminated)
                )

                obs = next_obs
                episode_reward += reward
                steps += 1

                if len(replay) >= max(batch_size, learning_starts) and steps % 4 == 0:
                    batch = random.sample(replay, batch_size)
                    states, actions, rewards, next_states, terminals = zip(*batch)

                    states = torch.as_tensor(
                        np.stack(states), dtype=torch.float32, device=device
                    ) / 255.0
                    next_states = torch.as_tensor(
                        np.stack(next_states), dtype=torch.float32, device=device
                    ) / 255.0
                    actions = torch.tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    terminals = torch.tensor(terminals, dtype=torch.float32, device=device)

                    predicted_q = online(states).gather(1, actions[:, None]).squeeze(1)

                    with torch.no_grad():
                        next_q = target(next_states).max(dim=1).values
                        target_q = rewards + gamma * (1.0 - terminals) * next_q

                    loss = loss_fn(predicted_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    updates += 1
                    if updates % 250 == 0:
                        target.load_state_dict(online.state_dict())

                if terminated or truncated:
                    break

            print(
                f"Episode {episode:03d} | reward {episode_reward:6.2f} | "
                f"epsilon {epsilon:.3f} | updates {updates}"
            )

        torch.save(online.state_dict(), "dqn_highway.pth")
        print("Saved dqn_highway.pth")
    finally:
        env.close()


if __name__ == "__main__":
    train()