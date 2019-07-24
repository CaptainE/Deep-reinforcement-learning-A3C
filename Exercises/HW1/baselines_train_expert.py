import pybulletgym  # register PyBullet enviroments with open ai gym
import gym  # open ai gym
'''
This file will be used to generate the 'experts' and save them in 
'''
method = 'ppo2'
import baselines
import baselines.run
import os


def fout_(method, ev):
    out = './agent_zoo_baselines/expert_%s_%s.pkl' % (method, ev)
    return out


def load_expert(ev):
    out = fout_(method, ev)

    args = ['--alg', method,
            '--env', ev,
            '--num_timesteps', '0',
            '--load_path', out]
    model = baselines.run.main(args)
    class PolicyInner:
        def act(self, ob):
            action, _, _, _ = model.step(ob)
            return action

    pi = PolicyInner()
    return pi


def train_expert(ev, steps = "1e6"):
    out = fout_(method, ev)
    print("Training %s for %s steps; saving to: %s"%(ev, steps, out))
    os.environ["OPENAI_LOG_FORMAT"] = "stdout,csv,tensorboard"
    os.environ["OPENAI_LOGDIR"] = "logs"  # you might want to change this

    args = ['--alg', method,
            '--env', ev,
            '--num_timesteps', steps,
            '--save_path', out]
    baselines.run.main(args)


if __name__ == "__main__":
    print("Pre-training agents...")
    import baselines.run
    ev = 'Humanoid-v2'
    train_expert(ev, '1e6')

    # example code to visualize logs:
    from baselines.common import plot_util as pu
    results = pu.load_results('logs')
    import matplotlib.pyplot as plt
    import numpy as np
    r = results[0]
    plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
    plt.show()
