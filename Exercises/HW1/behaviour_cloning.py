
"""
Train a BC agent using expert data in a given environment by running:
    python run_expert.py --expert_data expert_data/InvertedPendulumPyBulletEnv-v0.pkl --envname InvertedPendulumPyBulletEnv-v0

@author Peter & Nicklas
"""

import tensorflow as tf
import numpy as np
from random import shuffle
import pickle
import argparse
import gym
import pybulletgym
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_policy(env, pi, num_rollouts, max_steps, render=False, return_reward=False, verbose=False):
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

            if env.action_space.shape[0] > 1:
                action = action[0]

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

    if verbose:
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    if return_reward:
        return np.mean(returns), np.std(returns)
    
    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    return expert_data


class Agent(object):
    def __init__(self, env, n_input, n_output):

        # Start a new session
        self.sess = tf.Session()
        self.env = env

        # Create model            
        self.input_ph, self.output_ph, self.output_pred = self.create_model(n_input, n_output)

        # Define loss and optimizer
        self.mse = tf.reduce_mean(0.5 * tf.square((self.output_pred-self.output_ph)))
        self.opt = tf.train.AdamOptimizer().minimize(self.mse)


    def create_model(self, n_input, n_output):
        # Create inputs
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        output_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_output])

        # Create variables
        n_neurons1, n_neurons2 = 128, 128
        W0 = tf.get_variable(name='W0', shape=[n_input, n_neurons1], initializer=tf.contrib.layers.xavier_initializer())
        W1 = tf.get_variable(name='W1', shape=[n_neurons1, n_neurons2], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(name='W2', shape=[n_neurons2, n_output], initializer=tf.contrib.layers.xavier_initializer())
        
        b0 = tf.get_variable(name='b0', shape=[n_neurons1], initializer=tf.constant_initializer(0.))
        b1 = tf.get_variable(name='b1', shape=[n_neurons2], initializer=tf.constant_initializer(0.))
        b2 = tf.get_variable(name='b2', shape=[n_output], initializer=tf.constant_initializer(0.))

        # Create computation graph
        layer = input_ph
        layer = tf.nn.relu(tf.matmul(layer, W0) + b0)
        layer = tf.nn.relu(tf.matmul(layer, W1) + b1)
        layer = tf.matmul(layer, W2) + b2

        return input_ph, output_ph, layer


    def train(self, S_train, A_train, S_val, A_val, num_iterations=10000):
        # Define number of batches
        batch_size = 64

        # Initialize session
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        losses, rewards = [], []

        for iteration in range(1, num_iterations+1):
            # Sample a random batch
            indices = np.random.randint(low=0, high=len(S_train), size=batch_size)
            input_batch = S_train[indices]
            output_batch = A_train[indices]
            
            # Run optimizer and get MSE
            _, mse_run = self.sess.run([self.opt, self.mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
            losses.append(mse_run)

            # Save reward for plots
            if iteration % 10 == 0:
                reward = evaluate_policy(self.env, self, num_rollouts=10, max_steps=1000, return_reward=True)
                rewards.append(reward)

            # Print MSE every now and then
            if iteration % 500 == 0:
                mean, std = reward
                print(f'Iteration {iteration}, MSE: {mse_run}, Mean Reward: {mean}, Reward std: {std}')
                self.saver.save(self.sess, '/tmp/model.ckpt')

        return losses, rewards


    def act(self, s):
        s = np.reshape(s, (1, s.size))
        return self.sess.run(self.output_pred, feed_dict={self.input_ph: s})
        

def load_data(fname):
    # Load data from file
    with open(fname, 'rb') as f:
        expert_data = pickle.load(f)
    S = expert_data['observations']
    A = expert_data['actions']
    print('State shape:', S.shape)
    print('Action shape', A.shape)

    # Shuffle samples
    samples = list(zip(S, A))
    shuffle(samples)
    S, A = zip(*samples)
    S, A = np.array(S), np.array(A)

    # Partition dataset
    split_index = int(args.split_ratio*len(S))
    S_train, S_val = S[:split_index], S[split_index:]
    A_train, A_val = A[:split_index], A[split_index:]

    print(f'Train: {len(S_train)}, Test: {len(S_val)}')
    return S_train, S_val, A_train, A_val


def plot_loss_reward(losses, rewards, envname):
    def mv_avg(x, n=100):
        return np.convolve(x, np.ones((n,))/n, mode='valid')

    def plot_fig(x, std=None, is_loss = True):
        iterations = np.linspace(0, args.num_iterations, num=len(x))
        plt.style.use('seaborn-white')
        plt.figure(figsize=(8,6))
        plt.plot(iterations, x)
        if std is not None:
            plt.fill_between(iterations, x-std, x+std, alpha=0.2)
        plt.title(f'Behaviour Cloning ({envname})')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)' if is_loss else 'Mean Reward')
        plt.show()

    rewards_mean, rewards_std = zip(*rewards)
    losses, rewards_mean, rewards_std = mv_avg(losses), mv_avg(rewards_mean, n=10), mv_avg(rewards_std, n=10)
    plot_fig(losses)
    plot_fig(rewards_mean, std=rewards_std, is_loss=False)


def policy_imitation(args):
    # Load environment
    env = gym.make(args.envname)
    n_input = env.observation_space.shape[0]
    n_output = env.action_space.shape[0]

    # Load expert data
    S_train, S_val, A_train, A_val = load_data('expert_data/' + args.envname +'.pkl')

    # Train our policy
    pi = Agent(env, n_input, n_output)
    losses, rewards = pi.train(S_train, A_train, S_val, A_val, num_iterations=args.num_iterations)

    # Plot loss and reward
    plot_loss_reward(losses, rewards, args.envname)

    evaluate_policy(env, pi, num_rollouts=10, max_steps=env.spec.timestep_limit, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--envname', default="InvertedPendulumPyBulletEnv-v0", type=str)
    #parser.add_argument('--envname', default="HumanoidPyBulletEnv-v0", type=str)
    parser.add_argument('--envname', default="HalfCheetahPyBulletEnv-v0", type=str)
    parser.add_argument('--num_iterations', type=int, default=5000,
                        help='Number of training iterations')
    parser.add_argument('--split_ratio', type=float, default=1.0,
                        help='Ratio of test/train split')
    args = parser.parse_args()
    policy_imitation(args)
