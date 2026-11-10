import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

gym.register_envs(highway_env)

config = {
    "lanes_count": 5,
    "policy_frequency": 2,
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
}

env = gym.make("highway-v0", config=config, render_mode="human")

try:
    obs, _ = env.reset()
    idle_action = env.unwrapped.action_type.actions_indexes["IDLE"]
    print("Observation shape:", obs.shape)  # (4, 128, 64)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("Grayscale observations (oldest to newest)")
    images = []

    for ax, frame in zip(axes, obs):
        image = ax.imshow(frame.T, cmap="gray", vmin=0, vmax=255)
        images.append(image)
        ax.axis("off")

    plt.show(block=False)
    plt.pause(0.01)

    for step in range(1000):
        if not plt.fignum_exists(fig.number):
            break

        # Coast for three steps to fill the frame stack, then sample actions.
        action = idle_action if step < 3 else env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        for image, frame in zip(images, obs):
            image.set_data(frame.T)

        fig.canvas.draw_idle()
        plt.pause(0.01)

        if terminated or truncated:
            obs, _ = env.reset()

except KeyboardInterrupt:
    pass
finally:
    env.close()
    plt.close("all")