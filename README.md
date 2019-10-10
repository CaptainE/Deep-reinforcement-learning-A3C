# Deep-reinforcement-learning-A3C
Repository containing material regarding a modified version of the Berkeley Deep reinforcement learning course, that is it only contain some of the assignments for CS294-112, and a PyTorch implementation of Asynchronous Advantage Actor-Critic (A3C) using Generalized Advantage Estimation  as a project

Here only solutions material for homework 1, 4 and 5a is provided in Tensorflow.

The A3C algorithm is made possible by distributed learning in which numerous workers interact with the environment
and update model parameters asynchronously. (hence the name..) 
This removes the need for a memory buffer as with other algorithms, also the distributed learning allows for more 
efficient use of hardware as we can generate multiple rollouts running in parallel.

Since our rollouts are generated on-policy, there is a high chance that all trajectories end up being similar, 
as action probabilities gradually become near-zero for all but one action in the discrete case, which ultimately limits exploration. 
Thus we see another benefit of A3C as it addresses this problem by introducing an entropy-term to the loss function, which is discussed in section 2. 
We extend the A3C by replacing the advantage estimator used in (Minh, 2016) by the Generalized Advantage Estimate (GAE) as proposed by (Schulman, 2015), and evaluate the algorithm on a number of environments.

See our paper for more details.

The work presented in this repository is to be considered open source under the MIT License. If you found this code useful in your research, then please cite

```
@misc{hansen-ebert,
  title={Distributed Deep Reinforcement Learning with Asynchronous Advantage Actor-Critic using Generalized Advantage Estimation},
  author={Ebert, Peter and Hansen, Nicklas},
  year={2019}
}
```
