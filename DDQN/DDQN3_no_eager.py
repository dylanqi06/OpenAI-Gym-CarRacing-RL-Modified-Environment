# DDQN version 1 to improve on the performance of DQN2.py
# Best performance so far:
#   --> model/oah33/DDQN2/20220423-170444/episode_500.h5
#   --> NN input is greyscale 96x96x1

# Environment imports
import random
import numpy as np
import gym
import pyvirtualdisplay
import cv2
from scipy import stats

# Tensorflow training imports
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Training monitoring imports
import datetime, os
from tqdm import tqdm
import time
from plot_results import plotResults

# Disable Eager Execution
tf.compat.v1.disable_eager_execution()

# Create a TensorFlow session
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

############################## CONFIGURATION ##################################
# Prevent TensorFlow from allocating all of GPU memory
# From: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)   # set memory growth option

# Creates a virtual display for OpenAI gym (to support running from headless servers)
# pyvirtualdisplay.Display( visible=0, size=(720, 480) ).start()

# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "dylanqi"
MODEL_TYPE              = "DDQN3"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR               = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup Reward Dir
REWARD_DIR              = f"rewards/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Training params
RENDER                  = True
PLOT_RESULTS            = False     # plotting reward and epsilon vs episode (graphically) NOTE: THIS WILL PAUSE TRAINING AT PLOT EPISODE!
EPISODES                = 2000      # training episodes
SAVE_TRAINING_FREQUENCY = 100       # save model every n episodes
SKIP_FRAMES             = 2         # skip n frames between batches
TARGET_UPDATE_STEPS     = 5         # update target action value network every n EPISODES
MAX_PENALTY             = -30       # min score before env reset
BATCH_SIZE              = 20        # number for batch fitting
CONSECUTIVE_NEG_REWARD  = 25        # number of consecutive negative rewards before terminating episode
STEPS_ON_GRASS          = 20        # How many steps can car be on grass for (steps == states)
REPLAY_BUFFER_MAX_SIZE  = 10000     # threshold memory limit for replay buffer

# Testing params
PRETRAINED_PATH         = "DDQN/model/oah33/DDQN3_NN/20220424-140943/episode_900.h5"
TEST                    = False     # true = testing, false = training

# Initialize global counter for grass detection
on_grass_counter = 0

############################## MAIN CODE BODY ##################################
class DDQN_Agent:
    def __init__(   self, 
                    action_space    = [
                    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #            Action Space Structure
                    (-1, 1,   0), (0, 1,   0), (1, 1,   0), #           (Steering, Gas, Break)
                    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range       -1~1     0~1     0~1
                    (-1, 0,   0), (0, 0,   0), (1, 0,   0)],  
                    gamma           = 0.95,      # discount rate
                    epsilon         = 1.0,       # exploration rate
                    epsilon_min     = 0.1,       # used by Atari
                    epsilon_decay   = 0.9999,
                    learning_rate   = 0.001
                ):
        
        self.action_space    = action_space
        self.D               = deque(maxlen=REPLAY_BUFFER_MAX_SIZE)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        
        # Build the action value network and target network
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_model()  # Initialize target_model weights
        
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
    
    def build_model(self):
        """Sequential Neural Net with x2 Conv layers, x2 Dense layers using RELU and Mean Squared Error Loss"""
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', 
                      optimizer=Adam(learning_rate=self.learning_rate))  # Removed epsilon parameter
        return model

    def update_model(self):
        """Update Target Action Value Network to be equal to Action Value Network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def store_transition(self, state, action, reward, new_state, done):
        """Store transition in the replay memory (for replay buffer)."""
        self.D.append( (state, action, reward, new_state, done) )
    
    def choose_action(self, state, best=False, _random=False):
        """Take state input and use latest model to make prediction on best next action; choose it!"""
        state = np.expand_dims(state, axis=0).astype(np.float32)
        # Define input and output tensors
        input_tensor = self.model.input
        output_tensor = self.model.output
        # Run session to get Q-values
        q_values = sess.run(output_tensor, feed_dict={input_tensor: state})
        actionIDX = np.argmax(q_values[0])
        
        # Return best action if defined
        if best:
            return self.action_space[actionIDX]
        
        # Return random action
        if _random:
            return random.choice(self.action_space)
        
        # Epsilon chance to choose random action
        if stats.bernoulli(self.epsilon).rvs():
            actionIDX = random.randrange(len(self.action_space))
        return self.action_space[actionIDX]
    
    def batch_priority(self):
        """Implementation of prioritized replay where most recent states take highest priority."""
        options = list(range(1, len(self.D)+1))
        minibatch = []
        for _ in range(BATCH_SIZE):
            if not options:
                break
            # Update probs and create new distribution
            total = len(options) * (len(options) + 1) // 2
            prob_dist = [i/total for i in range(1, len(options)+1)]
            
            choice = np.random.choice(options, 1, p=prob_dist)[0]
            options.remove(choice)
            minibatch.append(self.D[choice-1])
        return minibatch
    
    def experience_replay(self):
        """Use experience_replay with batch fitting and epsilon decay."""
        if len(self.D) >= BATCH_SIZE:
            # Select batch based on prioritized replay approach
            minibatch = self.batch_priority()
            
            # Experience replay
            train_state = []
            train_target = []
            for state, action, reward, next_state, done in minibatch:
                # Get Q-values for current state
                q_values = sess.run(self.model.output, feed_dict={self.model.input: np.expand_dims(state, axis=0).astype(np.float32)})
                target = q_values[0]
                if done:
                    target[self.action_space.index(action)] = reward
                else:
                    # Double Deep Q Learning Here!
                    # Get index of action value network prediction for best action at next state
                    t = sess.run(self.model.output, feed_dict={self.model.input: np.expand_dims(next_state, axis=0).astype(np.float32)})
                    t_index = np.argmax(t[0])

                    # Get target network prediction for next state, then use index calc'd above to
                    # update Q action value network
                    target_t = sess.run(self.target_model.output, feed_dict={self.target_model.input: np.expand_dims(next_state, axis=0).astype(np.float32)})
                    target[self.action_space.index(action)] = reward + self.gamma * target_t[t_index]

                train_state.append(state)
                train_target.append(target)
            
            # Convert to numpy arrays
            train_state = np.array(train_state).astype(np.float32)
            train_target = np.array(train_target).astype(np.float32)
            
            # Batch fitting
            self.model.fit(train_state, train_target, epochs=1, verbose=0)
            
            # Epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self, name, data):
        """Save model and rewards list to appropriate dir, defined at start of code."""
        # Saving model
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # Use model.save_weights for compatibility
        self.model.save_weights(MODEL_DIR + name + ".weights.h5")
    
        # Saving results
        if not os.path.exists(REWARD_DIR):
            os.makedirs(REWARD_DIR)
        np.savetxt(f"{REWARD_DIR}" + name + ".csv", data, delimiter=",")
        
        # Plotting results
        if PLOT_RESULTS:
            plotResults(f"{REWARD_DIR}" + name + ".csv")
    
    def load(self, name):
        """Load previously trained model weights."""
        self.model.load_weights(name)
        self.target_model.set_weights(self.model.get_weights())

def convert_greyscale(state):
    """Take input state and convert to greyscale. Check if road is visible in frame."""
    global on_grass_counter
    
    if isinstance(state, tuple):
        state = state[0]
    x, y, _ = state.shape
    cropped = state[0:int(0.85*y), 0:x]
    mask = cv2.inRange(cropped,  np.array([100, 100, 100]),  # dark_grey
                              np.array([150, 150, 150]))  # light_grey
    
    # Create greyscale then normalize array to reduce complexity for neural network
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(float)
    gray_normalised = gray / 255.0
    
    # Check if car is on grass
    xc = int(x / 2)
    grass_mask = cv2.inRange(state[67:76, xc-2:xc+2],
                             np.array([50, 180, 0]),
                             np.array([150, 255, 255]))
    
    # If on grass for STEPS_ON_GRASS frames or more, trigger True
    if np.any(grass_mask == 255):
        on_grass_counter += 1
    else:
        on_grass_counter = 0
    
    on_grass = on_grass_counter > STEPS_ON_GRASS
    
    # Returns [ greyscale image, T/F of if road is visible, is car on grass bool ]
    return [np.expand_dims(gray_normalised, axis=2), np.any(mask == 255), on_grass]

def train_agent(agent: DDQN_Agent, env: gym.Env, episodes: int):
    """Train agent with experience replay, batch fitting and using a cropped greyscale input image."""
    episode_rewards = []
    for episode in tqdm(range(episodes)):
        print(f"[INFO]: Starting Episode {episode}")
        
        state_colour = env.reset()
        state_grey, can_see_road, car_on_grass = convert_greyscale(state_colour)

        sum_reward = 0
        step = 0
        done = False
        repeat_neg_reward = 0  # Initialize here
        while not done and sum_reward > MAX_PENALTY and can_see_road and not car_on_grass:
            # Choose action to take next
            action = agent.choose_action(state_grey)

            # Take action and observe new state, reward and if terminal.
            # Include "future thinking" by forcing agent to do chosen action 
            # SKIP_FRAMES times in a row. 
            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                new_state_colour, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                reward += r

                # Render if user has specified, break if terminal
                if RENDER:
                    env.render()
                if done:
                    break

            # Count number of negative rewards collected sequentially, if reward negative, increment
            if reward < 0:
                repeat_neg_reward += 1
            else:
                repeat_neg_reward = 0

            if repeat_neg_reward >= CONSECUTIVE_NEG_REWARD:
                break

            # Convert to greyscale for NN
            new_state_grey, can_see_road, car_on_grass = convert_greyscale(new_state_colour)

            # Clip reward to range [-10, 1]
            reward = np.clip(reward, a_max=1, a_min=-10)

            # Store transition states for experience replay
            agent.store_transition(state_grey, action, reward, new_state_grey, done)

            # Do experience replay training with a batch of data
            agent.experience_replay()

            # Update params for next loop
            state_grey = new_state_grey
            sum_reward += reward
            step += 1

        # Store episode reward
        episode_rewards.append([sum_reward, agent.epsilon])

        # Update target action value network every N episodes
        if episode % TARGET_UPDATE_STEPS == 0:
            agent.update_model()

        # Save model and rewards periodically
        if episode % SAVE_TRAINING_FREQUENCY == 0:
            agent.save(f"episode_{episode}", data=episode_rewards)
    env.close()

def test_agent(agent: DDQN_Agent, env: gym.Env, model: str, testnum=10):
    """Test a pretrained model and print out run rewards and total time taken. Quit with ctrl+c."""
    # Load agent model
    agent.load(model)
    run_rewards = []
    for test in range(testnum):
        state_colour = env.reset()
        state_grey, _, _ = convert_greyscale(state_colour)

        done = False
        sum_reward = 0.0
        repeat_neg_reward = 0  # Initialize here
        t_start = time.time()  # Trial timer
        while sum_reward > MAX_PENALTY and not done:
            # Choose action to take next
            action = agent.choose_action(state_grey, best=True)
            
            # Take action and observe new state, reward and if terminal
            new_state_colour, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Render if user has specified
            if RENDER:
                env.render()

            # Count number of negative rewards collected sequentially
            if reward < 0:
                repeat_neg_reward += 1
            else:
                repeat_neg_reward = 0

            if repeat_neg_reward >= 300:
                break

            # Convert to greyscale for NN
            new_state_grey, _, _ = convert_greyscale(new_state_colour)

            # Update state
            state_grey = new_state_grey
            sum_reward += reward

        elapsed_time = time.time() - t_start
        run_rewards.append([sum_reward, np.nan, elapsed_time, np.nan, np.nan, np.nan])
        print(f"[INFO]: Run {test} | Run Reward: {sum_reward} | Time: {elapsed_time:.2f}s.")

    # Calculate useful statistics
    rr = [i[0] for i in run_rewards]
    rt = [i[2] for i in run_rewards]

    r_max = max(rr)
    r_min = min(rr)
    r_std_dev = np.std(rr)
    r_avg = np.mean(rr)
    t_avg = np.mean(rt)
    
    run_rewards.append([r_avg, np.nan, t_avg, r_max, r_min, r_std_dev])    # STORE AVG RESULTS AS LAST ENTRY!
    print(f"[INFO]: Runs {testnum} | Avg Run Reward: {r_avg:.2f} | Avg Time: {t_avg:.2f}s | Max: {r_max} | Min: {r_min} | Std Dev: {r_std_dev}")

    # Saving test results
    test_reward_dir = f"test_{REWARD_DIR}"
    if not os.path.exists(test_reward_dir):
        os.makedirs(test_reward_dir)
    path = f"{test_reward_dir}" + PRETRAINED_PATH.split('/')[-1][:-3] + "_run_rewards.csv"
    np.savetxt(path, run_rewards, delimiter=",")

    # Return average results
    return [r_avg, np.nan, t_avg, r_max, r_min, r_std_dev]

if __name__ == "__main__":
    # Initialize the CarRacing environment
    env = gym.make('CarRacing-v2', render_mode='rgb_array').env

    if not TEST:
        # Train Agent
        agent = DDQN_Agent()
        train_agent(agent, env, episodes=EPISODES)
    
    else:
        # Test Agent
        agent = DDQN_Agent()
        test_agent(agent, env, model=PRETRAINED_PATH, testnum=50)

    # Close the TensorFlow session
    sess.close()
