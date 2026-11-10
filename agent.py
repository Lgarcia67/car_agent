import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, num_actions=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state):
        return self.network(state)


if __name__ == "__main__":
    agent = Agent()
    state = torch.zeros((1, 4, 128, 64), dtype=torch.float32)

    with torch.no_grad():
        q_values = agent(state)

    print("Output shape:", q_values.shape)  # torch.Size([1, 5])
    print("Q-values:", q_values)
    print("Selected action:", q_values.argmax(dim=1).item())