# Overview
This repository contains code for various reinforcement learning algorithms such as DQN, Double DQN and DDPG. Click on each of their sub-folders to find out specific information on how they each work. See below for an overview on the project results.


![DQN, DDQN](/imgs/results.png "Discrete Action Space RL Approaches")


*One dense layer more and more _"lenient episode termination"_ (meaning the car can drive on the grass for longer before terminating episode)


# Building Conda Env on Local Machine
A yml script has been created to help with building your anaconda environment. Simply run the command below in terminal (which supports anaconda). Environment name can be changed from SERVER_ENV to user choice inside yml file.
```shell
conda env create --file build_conda_env.yml
```

# Server Setup

## In Code (the model to train)
Tensorflow, by default, hogs/ allocates all the GPU memory. This is *NOT* good as this often leads to *Out of Memory* errors during training. In order to prevent this, add the following lines of code at the *TOP* of your script. Essentially, this will enforce that Tensorflow only allocates as much memory as is needed at that given time.
```python
# Prevent tensorflow from allocating the all of GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)   # set 
```

Open AI Gym uses pyglet which requires a screen/ monitor to function. The servers don't have said screens so we make virtual ones within our code to spoof the code into thinking we do. See below:
```python
import pyvirtualdisplay

# Creates a virtual display for OpenAI gym
pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
```
