import tensorflow as tf
import numpy as np
import tf_util
import gym
import pybulletgym
import matplotlib.pyplot as plt
from random import shuffle
import argparse
import pickle
from agent_zoo_pybulletgym.enjoy_TF import enjoy_TF


def dagger(args):
    '''
    Import dagger here. Use evaluate_policy (and code within) to get an idea of how to interact
    with the environment. Note you can re-use the model you implemented for behavioural cloning.

    To load the (expert), use

    from agent_zoo_pybulletgym.enjoy_TF import enjoy_TF
    ev = 'InvertedPendulumPyBulletEnv-v0'
    pic = enjoy_TF(ev)
    env=gym.make(ev)
    expertdata_data=evaluate_policy(env,pic,1)

    '''

    memory =[]
    # Load environment
    env = gym.make(args.envname)
    n_input = env.observation_space.shape[0]
    n_output = env.action_space.shape[0]

    # Load expert data
    
    pic = enjoy_TF(args.envname)
    ourdata= evaluate_policy(env, pic, num_rollouts=200, max_steps=1000, render=False, verbose=True)
    S=ourdata['observations']
    A=pic.act(S)
    print(A.shape)
    print(S.shape)
    
    #S,A = load_data('expert_data/' + args.envname +'.pkl')
    memory= list(zip(np.array(S),np.array(A)))
    pi = Agent(env, n_input, n_output, memory)
    # Initialize session
    pi.sess.run(tf.global_variables_initializer())
    reward_list = []
    reward_list_bc = []
    for i in range(args.num_dagger_iter):
        
        #graph2=tf.Graph()
        #with graph2.as_default():
        #    bcAgent = Agent(gym.make(args.envname), n_input, n_output, memory)
        #    bcAgent.sess.run(tf.global_variables_initializer())
        #step 1
        pi.memory=memory
        #bcAgent.memory=memory
        S,A = zip(*pi.memory)
        losses, rewards = pi.train(num_iterations=args.num_iterations)
        #losses, rewards = bcAgent.train(num_iterations=args.num_iterations)

        #step 2
        ourdata= evaluate_policy(env, pi, num_rollouts=10, max_steps=1000, render=False, verbose=True)

        mean,std= evaluate_policy(env, pi, num_rollouts=10, max_steps=1000, render=False, return_reward=True, verbose=False)
        reward_list.append([mean,std])
        #mean,std= evaluate_policy(env, bcAgent, num_rollouts=10, max_steps=1000, render=False, return_reward=True, verbose=False)
        #reward_list_bc.append([mean,std])
		
        #step3
        states=ourdata['observations']
        pic = enjoy_TF(args.envname)
        actions=pic.act(states)
        
        #step 4
        S,A = zip(*memory)
        S=list(S)+list(states)
        A=list(A)+list(actions)
        S,A=np.array(S),np.array(A)
        memory= list(zip(S,A))
    plot_loss_reward(reward_list, args.envname)
    evaluate_policy(env, pi, num_rollouts=10, max_steps=env.spec.timestep_limit, render=False, verbose=True)

def evaluate_policy(env, pi, num_rollouts, max_steps, render=False, return_reward=False, verbose=False):
    global checker
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
            if env.action_space.shape[0] > 1 and len(action.shape)>1:
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
    def __init__(self, env, n_input, n_output,memory):

        # Start a new session
        self.sess = tf.Session()
        self.env = env

        # Create model            
        self.input_ph, self.output_ph, self.output_pred = self.create_model(n_input, n_output)
        
        #Memory
        self.memory=memory
        

        # Define loss and optimizer
        self.mse = tf.reduce_mean(0.5 * tf.square((self.output_pred-self.output_ph)))
        self.opt = tf.train.AdamOptimizer().minimize(self.mse)


    def create_model(self, n_input, n_output):
        # Create inputs
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        output_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_output])

        # Create variables
        n_neurons1, n_neurons2, n_neurons3 = 256, 128, 64

        W0 = tf.get_variable(name='W0', shape=[n_input, n_neurons1], initializer=tf.contrib.layers.xavier_initializer())
        W1 = tf.get_variable(name='W1', shape=[n_neurons1, n_neurons2], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(name='W2', shape=[n_neurons2, n_neurons3], initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable(name='W3', shape=[n_neurons3, n_output], initializer=tf.contrib.layers.xavier_initializer())
        
        b0 = tf.get_variable(name='b0', shape=[n_neurons1], initializer=tf.constant_initializer(0.))
        b1 = tf.get_variable(name='b1', shape=[n_neurons2], initializer=tf.constant_initializer(0.))
        b2 = tf.get_variable(name='b2', shape=[n_neurons3], initializer=tf.constant_initializer(0.))
        b3 = tf.get_variable(name='b3', shape=[n_output], initializer=tf.constant_initializer(0.))

        # Create computation graph
        layer = input_ph
        layer = tf.nn.relu(tf.matmul(layer, W0) + b0)
        layer = tf.nn.relu(tf.matmul(layer, W1) + b1)
        layer = tf.nn.relu(tf.matmul(layer, W2) + b2)
        layer = tf.matmul(layer, W3) + b3
        return input_ph, output_ph, layer


    def train(self, num_iterations=5000):
        shuffle(self.memory)
        S,A = zip(*self.memory)
    
        
        S,A=np.array(S),np.array(A)
        # Partition dataset
        split_index = int(args.split_ratio*len(S))
        S_train, S_val = S[:split_index], S[split_index:]
        A_train, A_val = A[:split_index], A[split_index:]
        
        # Define number of batches
        batch_size = 64
        
        #self.saver = tf.train.Saver()
        losses, rewards = [], []

        for iteration in range(num_iterations):
            # Sample a random batch
            indices = np.random.randint(low=0, high=len(S_train), size=batch_size)
            input_batch = S_train[indices]
            output_batch = A_train[indices]
            
            # Run optimizer and get MSE
            _, mse_run = self.sess.run([self.opt, self.mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
            losses.append(mse_run)

            # Save reward for plots
            if iteration % 10 == 0:
                reward = evaluate_policy(self.env, self, num_rollouts=5, max_steps=1000, return_reward=True)
                rewards.append(reward)

            # Print MSE every now and then
            if iteration % 500 == 0:
                mean,std = reward
                print(f'Iteration {iteration}, MSE: {mse_run}, Mean Reward: {mean}, Reward std: {std}')
                #self.saver.save(self.sess, '/tmp/model.ckpt')

        return losses, rewards


    def act(self, s):
        s = np.reshape(s, (1, s.size))
        return self.sess.run(self.output_pred, feed_dict={self.input_ph: s})

def plot_loss_reward(rewards, envname):

    def plot_fig(x,std=None, is_loss = True):
        iterations = np.linspace(0, args.num_dagger_iter, num=len(x))
        plt.style.use('seaborn-white')
        plt.figure(figsize=(8,6))
        plt.plot(iterations, x)
        if std is not None:
            plt.fill_between(iterations, x-std, x+std, alpha=0.2)
        plt.title(f'DAgger ({envname})')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)' if is_loss else 'Mean Reward')
        plt.show()
    print(rewards)
    rewards_mean, rewards_std = zip(*rewards)
    
    plot_fig(np.array(rewards_mean), std=np.array(rewards_std), is_loss=False)

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
    A=list(np.squeeze(np.array(A),axis=1))
    S, A = np.array(S), np.array(A)

    return S,A
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_data', default="expert_data/InvertedPendulumPyBulletEnv-v0.pkl", type=str)
    parser.add_argument('--envname', default="InvertedPendulumPyBulletEnv-v0", type=str)
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--num_dagger_iter',type=int, default=10,
                        help='Number of DAgger iterations')
    parser.add_argument('--split_ratio', type=float, default=1.0,
                        help='Ratio of test/train split')
    args = parser.parse_args()

    dagger(args)