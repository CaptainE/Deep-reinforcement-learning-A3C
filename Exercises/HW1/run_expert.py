#!/usr/bin/env python
#     python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 \
#             --num_rollouts 20
"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
To load one of the pre-supplied (static) policies use:
    python run_expert.py expert_data/InvertedPendulumPyBulletEnv-v0 InvertedPendulumPyBulletEnv-v0 --num_rollouts 20

To use one of the policies you train using the baselines:
    python run_expert.py expert_data/InvertedPendulumPyBulletEnv-v0 InvertedPendulumPyBulletEnv-v0 --use_baselines True --num_rollouts 20


Check the code to see what this flag does in practice :-).

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
# import tensorflow as tf
import numpy as np
# import tf_util
import gym
import pybulletgym
import baselines_train_expert
from agent_zoo_pybulletgym.enjoy_TF import enjoy_TF

def evaluate_policy(env, pi, num_rollouts, max_steps, render=False):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = pi.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)
        if render:
            break

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    return expert_data

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('expert_file_out', default=None, type=str, help="Which file to save results to")
    parser.add_argument('envname', type=str)
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('--roboschool', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--use_baselines', type=bool, default=False,
                        help='Use a pre-trained policy included by pybulletgym')
    args = parser.parse_args()

    if args.expert_file_out is None:
        fname = os.path.join('expert_data', args.envname + '.pkl')
    else:
        fname = args.expert_file_out


    env = gym.make(args.envname)

    print('loading and building expert policy')
    if args.use_baselines:
        pi = enjoy_TF(args.envname)
    else:
        pi = baselines_train_expert.load_expert(args.envname)
    print('loaded and built')

    expert_data = evaluate_policy(env, pi, args.num_rollouts, args.max_timesteps, render=False)

    # with tf.Session():
    #     tf_util.initialize()
    #     max_steps = args.max_timesteps
    #
    #     returns = []
    #     observations = []
    #     actions = []
    #     print(args.num_rollouts)
    #     for i in range(args.num_rollouts):
    #         print('iter', i)
    #         obs = env.reset()
    #         done = False
    #         totalr = 0.
    #         steps = 0
    #         while not done:
    #             action = pi.act(obs)
    #             observations.append(obs)
    #             actions.append(action)
    #             obs, r, done, _ = env.step(action)
    #             totalr += r
    #             steps += 1
    #             if args.render:
    #                 env.render()
    #             if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
    #             if steps >= max_steps:
    #                 break
    #         returns.append(totalr)
    #         if args.render:
    #             break
    #
    #     print('returns', returns)
    #     print('mean return', np.mean(returns))
    #     print('std of return', np.std(returns))
    #
    #     expert_data = {'observations': np.array(observations),
    #                    'actions': np.array(actions)}

    with open(fname, 'wb') as f:
        print('Dumping simulations of expert policy to %s'%fname)
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
