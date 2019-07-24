# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:43:40 2019

@author: Peter
"""

import tensorflow as tf
from tensorflow import keras
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    
    with open(path, 'rb') as f:
        expert_data = pickle.load(f)
        
    return expert_data



def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.nn.relu(out_layer)
    return out_layer


def main():
    batch_size=100
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    import gym
    import pybulletgym
    env = gym.make(args.envname)
    data = load_data('expert_data/'+args.expert_data)

    xdata = data['observations']
    ydata = data['actions']
    ydata=list(np.squeeze(np.array(ydata),axis=1))

    ratio = 0.2
    split_index = int(ratio*len(xdata))
    xdata_train=xdata[split_index:]
    train_n = len(xdata_train)
    ydata_train=ydata[split_index:]
    xdata_valid=xdata[:split_index]
    ydata_valid=ydata[:split_index]
    
    n_input=env.observation_space.shape[0]
    n_output=env.action_space.shape[0]
    weights = {
    'h1': tf.Variable(tf.random_normal([n_input, 256])),
    'h2': tf.Variable(tf.random_normal([256, 128])),
    'h3': tf.Variable(tf.random_normal([128, 64])),
    'out': tf.Variable(tf.random_normal([64, n_output]))
    }

    biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([128])),
    'b3': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_output]))
    }
    
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    output=multilayer_perceptron(x,weights,biases)
    
    cost = tf.reduce_mean(tf.square((output-y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    total_batch=int(train_n/batch_size)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(args.num_epochs):
            avg_cost = 0.0
            x_batches = np.array_split(xdata_train, total_batch)
            y_batches = np.array_split(ydata_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                
                _, c = sess.run([optimizer, cost], 
                                feed_dict={
                                    x: batch_x, 
                                    y: batch_y
                                })
                avg_cost += c / total_batch
            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            
            x_valid = np.array(xdata_valid)
            y_valid = np.array(ydata_valid)

            _loss = sess.run(cost, feed_dict={x: x_valid, y: y_valid})
        
            print("[Test][Epoch: {}][Loss: {}]".format(epoch, _loss))
        print("\nFinish Training ...\n")
		
    	# testing
        env = gym.make(args.envname)
        max_steps = env.spec.timestep_limit

        returns = []
        states = []
        actions = []

        for i in range(10):
            print('iteration: ', i)
            state = env.reset()
            stopEnv = False
            sumofR, steps = 0,0
            while not stopEnv:
                state = np.array([state])
                action = sess.run(output, feed_dict={x: state})[0]
                states.append(state)
                actions.append(action)
                state, r, stopEnv, _ = env.step(action)
                sumofR += r
                steps += 1
                env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(sumofR)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
    return print('done')
	
if __name__ == '__main__':
    main()
