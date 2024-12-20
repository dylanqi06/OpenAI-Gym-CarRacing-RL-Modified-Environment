# Test multiple agent models at once and store results in csv file format

# Environment imports
import numpy as np
import gym

# Training monitoring imports
import datetime, os
from tqdm import tqdm
import time


############################## SERVER CONFIGURATION ##################################
# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "oah33"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Training params
RENDER                  = False
MAX_PENALTY             = -30       # min score before env reset


def test_agent( name : str, AgentClass, togray,  env : gym.make, model : list, testnum=10 ):
    """Test a pretrained model and print out run rewards and total time taken. Quit with ctrl+c."""
    # initialize the class
    agent = AgentClass()
    # model_ep = int(model.split('/')[-1][:-3].split("_")[-1])
    model_ep = int(model.split('/')[-1][:-11].split("_")[-1])


    # Load agent model
    agent.load( model )
    run_rewards = []
    for test in range(testnum):
        state_colour = env.reset() 
        state_grey, _, _ = togray( state_colour )

        done = False
        sum_reward = 0.0
        t1 = time.time()  # Trial timer
        while sum_reward > MAX_PENALTY and not done:

            # choose action to take next
            action = agent.choose_action( state_grey, best=True )
            
            # take action and observe new state, reward and if terminal
            # new_state_colour, reward, done, _ = env.step( action )
            new_state_colour, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # render if user has specified
            if RENDER: env.render()

            # Count number of negative rewards collected sequentially, if reward non-negative, restart counting
            repeat_neg_reward = repeat_neg_reward+1 if reward < 0 else 0
            if repeat_neg_reward >= 300: break

            # convert to greyscale for NN
            new_state_grey, _, _ = togray( new_state_colour )

            # update state
            state_grey = new_state_grey
            sum_reward += reward

        t1 = time.time()-t1
        run_rewards.append( [sum_reward, np.nan, t1, np.nan, np.nan, np.nan] )

    # calculate useful statistics
    rr = [ i[0] for i in run_rewards ]
    rt = [ i[2] for i in run_rewards ]

    r_max = max(rr)
    r_min = min(rr)
    r_std_dev = np.std( rr )
    r_avg = np.mean(rr)
    t_avg = np.mean(rt)
    
    print(f"[INFO]: Agent | {name} | Episode: {model_ep} | Runs {testnum} | Avg Run Reward: ", "%0.2f"%r_avg, "| Avg Time:", "%0.2fs"%t_avg,
            f" | Max: {r_max} | Min: {r_min} | Std Dev: {r_std_dev}" )

    # return average results
    return [r_avg, np.nan, t_avg, r_max, r_min, r_std_dev]


if __name__ == "__main__":
    # Import agents

    # from DDQN2 import DDQN_Agent as DDQN_Agent2
    # from DDQN2 import convert_greyscale as DDQN2_convert_greyscale
    from DDQN2_action import DDQN_Agent as DDQN_Agent2
    from DDQN2_action import convert_greyscale as DDQN2_convert_greyscale


    agents_functs_folders = [ ["DDQN2", DDQN_Agent2, DDQN2_convert_greyscale, "model/dylanqi/DDQN2/20241205-010828"], # two for no mod
                            # ["DDQN2", DDQN_Agent2, DDQN2_convert_greyscale, "model/dylanqi/DDQN2/20241129-123456"], # 3 for color mod
                            #  ["DDQN2", DDQN_Agent2, DDQN2_convert_greyscale, "model/dylanqi/DDQN2/20241128-122649"], # all for color mod
                            #  ["DDQN2", DDQN_Agent2, DDQN2_convert_greyscale, "model/dylanqi/DDQN2/20241127-123456"], # two for no mod
                            # ["DDQN2", DDQN_Agent2, DDQN2_convert_greyscale, "model/dylanqi/DDQN2/20241123-005953"],  # all for no mod             
                            ]



    # env = gym.make('CarRacing-v0').env
    env = gym.make('CarRacing-v2', render_mode='rgb_array').env
    # env = gym.make('CarRacing-v2', render_mode='rgb_array',domain_randomize=True).env

    for name, agent, grayscale_funct, folder in agents_functs_folders:
        print("\n\n [INFO]: NEW MODEL TYPE!")
        avg_runs = []
        for curr_model in os.listdir( folder ):
            if curr_model.endswith(".weights.h5"):
                episode = int(curr_model[:-11].split("_")[-1])
                # if episode % 100 == 0:
                if episode == 1400:
                    print("\n\n [INFO]: NEW EPISODE!")

                    # perform agent model testing
                    r_avg, _, t_avg, r_max, r_min, r_std_dev = test_agent(  name=name,
                                                                            AgentClass=agent,
                                                                            togray=grayscale_funct,
                                                                            env=env,
                                                                            model = folder + "/" + curr_model,
                                                                            testnum=50
                                                                        )

                    # append results to array
                    # episode = int(curr_model.split("_")[1][:-3])
                    avg_runs.append( [episode, r_avg, _, t_avg, r_max, r_min, r_std_dev] )

        # sort list into numerical order (ascending)
        avg_runs.sort(key=lambda x: x[0])

        # saving test results
        if not os.path.exists( f"episode_test_runs/{USERNAME}/{TIMESTAMP}/{name}/" ):
                os.makedirs( f"episode_test_runs/{USERNAME}/{TIMESTAMP}/{name}/" )
        path = f"episode_test_runs/{USERNAME}/{TIMESTAMP}/{name}/episode_run_rewards.csv"
        np.savetxt( path , avg_runs, delimiter=",")

    print("[SUCCESS]: Testing Done!!")