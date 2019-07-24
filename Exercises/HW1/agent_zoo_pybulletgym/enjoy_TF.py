import gym
import pybulletgym
import glob
import importlib
import os
dirn = os.path.dirname( os.path.realpath(__file__) )

def enjoy_TF(ev):
    env = gym.make(ev)
    ls = glob.glob("%s/*%s*"%(dirn, ev[:-3]) ).pop()[:-3]
    ls = os.path.basename(ls)
    p = importlib.import_module('agent_zoo_pybulletgym.%s'%ls)
    srp = p.SmallReactivePolicy(env.observation_space, env.action_space)
    return srp

if __name__ == "__main__":
    enjoy_TF("InvertedPendulumPyBulletEnv-v0")