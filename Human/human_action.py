import gym
import numpy as np
import pyglet
from pyglet.window import key

# Global variables for action and control
a = np.array([0.0, 0.0, 0.0])  # Steering, Acceleration, Brake
bool_do_not_quit = True

# Handle key press
def key_press(k, mod):
    global a, bool_do_not_quit
    if k == key.ESCAPE: bool_do_not_quit = False
    if k == key.Q: bool_do_not_quit = False
    # if k == key.LEFT: a[0] = -1.0  # Steer left
    # if k == key.RIGHT: a[0] = +1.0  # Steer right
    # if k == key.UP: a[1] = +1.0  # Accelerate
    # if k == key.DOWN: a[2] = +0.8  # Brake
    if k == key.RIGHT: a[0] = -1.0  # Steer left
    if k == key.LEFT: a[0] = +1.0  # Steer right
    if k == key.DOWN: a[1] = +1.0  # Accelerate
    if k == key.UP: a[2] = +0.8  # Brake

# Handle key release
def key_release(k, mod):
    global a
    if k == key.RIGHT and a[0] == -1.0: a[0] = 0
    if k == key.LEFT and a[0] == +1.0: a[0] = 0
    if k == key.DOWN: a[1] = 0
    if k == key.UP: a[2] = 0

# Main function
def run_carRacing_asHuman():
    global bool_do_not_quit, a

    num_runs = 10  # Number of rounds to play
    rewards = []

    # Initialize the environment
    env = gym.make('CarRacing-v2', render_mode='rgb_array').env
    # env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=True).env
    state = env.reset()

    # Create a single Pyglet window
    window = pyglet.window.Window(
        width=env.render().shape[1],
        height=env.render().shape[0],
        caption="Car Racing"
    )
    window.push_handlers(on_key_press=key_press, on_key_release=key_release)

    @window.event
    def on_close():
        global bool_do_not_quit
        bool_do_not_quit = False
        window.close()

    current_run = 0
    total_reward = 0  # Initialize total reward for the current round
    done = False

    def update(dt):
        global bool_do_not_quit, a
        nonlocal current_run, total_reward, done, state

        if not bool_do_not_quit:
            pyglet.app.exit()
            return

        if done:  # If the round is finished
            rewards.append(total_reward)
            print(f"Run {current_run + 1}: Total Reward = {total_reward}")
            current_run += 1

            if current_run >= num_runs:  # If all runs are complete
                pyglet.app.exit()
                return

            # Reset the environment for the next round
            state = env.reset()
            total_reward = 0
            done = False
            return

        # Step the environment
        state, reward, terminated, truncated, _ = env.step(a)
        # state, reward, done, _, _ = env.step(a)
        done = terminated or truncated
        total_reward += reward
        img = env.render()

        # Convert the image for Pyglet rendering
        img = pyglet.image.ImageData(
            img.shape[1], img.shape[0], 'RGB', img.tobytes(), pitch=-img.shape[1] * 3
        )
        window.clear()
        img.blit(0, 0)

    # Schedule updates
    pyglet.clock.schedule_interval(update, 1 / 60.0)  # 60 FPS
    pyglet.app.run()

    # Close the environment after all runs are complete
    env.close()

    # Compute statistics
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    print("\n--- Final Results ---")
    print(f"Average Reward: {avg_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Min Reward: {min_reward}")

# Run the game
run_carRacing_asHuman()
