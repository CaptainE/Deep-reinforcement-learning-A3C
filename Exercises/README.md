# Exercises and project description
This page contain overview of all mandatory tasks.

# Syllabus

# Homeworks


## Homework 1 (HW1): Imitation learning
 - Deadline: Upload this no later than WEDNESDAY ..., 
 - All relevant files can be found in https://gitlab.gbar.dtu.dk/deepRLcourse/RL_public/tree/master/Exercises/HW1

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * Roboschool **1.0.46**
 * OpenAI Gym version **0.10.5**

<!--- * MuJoCo version **1.50** and mujoco-py **1.50.1.56** --->

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note** For installing Roboschool please follow the instructions [here](https://github.com/openai/roboschool)

**Note** Set this in a virtual environment or use the '''--user''' flag
<!---
%There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).
%**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.

%**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.
--->
The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

(TODO: Add additional hints, etc. here)

## Homework 2 (HW1)

## Homework 3 (HW1)

## Project work
 - Default project: Either take 4, 5a,b or c as a starting point, read relevant background material, complete tasks and one additional experiment/comparison inspired by material project is based upon. 
 - Apply algorithms to a novel (non-mujoco environment), e.g. Sonic or DOOM. Apply idea for exploration to the new environment. Focus is on comparison between methods (AC vs. DQN, no exploration vs. exploration) across multiple "levels". 
 - Propose your own project

## Week 1  (Project 1 and 2)
 - Monday-Wednesday: Install TF/Torch. Modification: Mujoco --> bullet. Tensorflow tutorial.  
  Lecture 1-4: Use these for project 1 (specify)
 - Wednesday-Friday: Lecture 5-6 (Actor-critic). Do HW2

## Week 2 (Project)
 - Monday-Friday: Q-learning delen (7 8). Lecture 9 (TRPO) is optional (don't read)
 - Wednesday (afternoon): Discussion about projects/project selection. Upload project idea/abstract Wednesday
 
## Week 3
