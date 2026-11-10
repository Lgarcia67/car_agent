
    
  
"""Evaluate a trained Highway DQN on a fixed set of traffic seeds."""

import argparse
from pathlib import Path
from statistics import mean


TARGET_REWARD = 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", nargs="?", type=Path, default=Path("dqn_highway.pth")
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000, help="First traffic seed.")
    parser.add_argument("--render", action="store_true", help="Show the driving window.")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    import torch

    from agent import Agent
    from highway import Highway

    highway = Highway()
    env = highway.env
    rewards = []
    crashes = 0

    try:
        # Let the test control when the drive ends.
        env.unwrapped.configure({"duration": float("inf"), "offroad_terminal": False})

        agent = Agent(num_actions=env.action_space.n)
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        agent.load_state_dict(weights)
        agent.eval()

        print(f"Testing {args.checkpoint} on {args.episodes} episode(s).")

        with torch.inference_mode():
            for seed in range(args.seed, args.seed + args.episodes):
                obs, _ = highway.reset(seed=seed)
                total_reward = 0.0
                steps = 0

                while True:
                    # Match train.py: add a batch dimension and scale pixels to [0, 1].
                    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0
                    action = agent(state).argmax(dim=1).item()
                    obs, reward, _, _, info = env.step(action)
                    total_reward += float(reward)
                    steps += 1

                    crashed = bool(info["crashed"])
                    if crashed or total_reward >= TARGET_REWARD:
                        break

                crashes += int(crashed)
                rewards.append(total_reward)
                outcome = "crash" if crashed else "1,000 points reached"
                print(
                    f"Seed {seed} | reward {total_reward:7.2f} | "
                    f"{steps:3d} steps | {outcome}"
                )

        print(f"Average reward: {mean(rewards):.2f}")
        print(f"Crashes: {crashes}/{args.episodes} ({crashes / args.episodes:.0%})")
    finally:
        env.close()


if __name__ == "__main__":
    main()

