from collections import deque
import os
import random
from tqdm import tqdm

import torch
from ipdb import set_trace
import numpy as np

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory, PERMemory, Memory_Buffer_PER
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--max_step', type=int, default=500000)
parser.add_argument('--lab_types_idx', type=str)
parser.add_argument('--count_shade', type=int, default=-1)
parser.add_argument('--evaluate_freq', type=int, default=5000)

args = parser.parse_args()


#os

GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 500_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 10000

BATCH_SIZE = 32

POLICY_UPDATE = 4
TARGET_UPDATE = 1_000
WARM_STEPS = 1000
MAX_STEPS = 500_000
STABLE_STEPS = MAX_STEPS // 2
EVALUATE_FREQ = 5_000

MAX_STEPS = args.max_step
LAB_IDX = [int(i) for i in args.lab_types_idx.split(',')]
#LAB_IDX = args.lab_types_idx
COUNT_SHADE = args.count_shade != -1
EVALUATE_FREQ = args.evaluate_freq

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
if not os.path.exists(SAVE_PREFIX):
    os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vs = ['natural', 'dueling', 'PER', 'dueling+PER', 'natural+stable']
co = {'natural':'r', 'dueling':'g', 'PER':'b', 'dueling+PER':'y', 'natural+stable':'c'}
vs0 = []

for id in LAB_IDX:
    print(id)
    vs0.append(vs[id])
#vs0 = vs[0:2] + vs[2:3]
avg_reward=0.0
#torch.cuda.set_device(2)

pretrained = './saved_models/model_weights_b'
#pretrained = None

action_queues = []

def get_shade(action_queue):
    cnt = 0
    for i in range(1, len(action_queue)-1):
        for j in [i-1, i+1]:
            cnt += 1*(action_queue[i]*action_queue[j]==2)
    return cnt/len(action_queue)

def count_shade(action_queues, versions):
    l = len(versions)
    for i in range(l):
        action_queue = action_queues[i]
        ve = versions[i]
        print('current lab:',ve,'shade rate:',get_shade_time(action_queue))

if __name__ == "__main__":
    
    versions = vs0
    for version in versions:
        #set_trace()
        print(version)
        dueling = False if version.find('dueling') == -1 else True
        stable = False if version.find('stable') == -1 else True
        if stable:
            action_queue = []
        env = MyEnv(device)
        agent = Agent(
            env.get_action_dim(),
            device,
            GAMMA,
            new_seed(),
            EPS_START,
            EPS_END,
            EPS_DECAY,
            dueling,
            pretrained,
            stable*0.1
        )
        if version.find('PER') != -1:
            memory = PERMemory(STACK_SIZE+1, MEM_SIZE, device) 
            #memory = Memory_Buffer_PER(MEM_SIZE)
        else:
            memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device) 
            #memory = Memory_Buffer_PER(MEM_SIZE)

        #### Training ####
        obs_queue: deque = deque(maxlen=5)
        done = True

        avg_reward_arr = []

        progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                        ncols=50, leave=False, unit="b")
        for step in progressive:
            if done:
                observations, _, _ = env.reset()
                for obs in observations:
                    obs_queue.append(obs)

            training = len(memory) > WARM_STEPS
            state = env.make_state(obs_queue).to(device).float()        #current state S
            action, value_this = agent.run(state, training)             #from state and DQN get Q-function, policy and action A
            if stable and COUNT_SHADE:
                action_queue.append(action)
            obs, reward, done = env.step(action)                        #execute action getting R, S'
            obs_queue.append(obs)
            if version.find('PER') != -1:
                state_next = env.make_state(obs_queue).to(device).float()
                value_next = agent.get_target_value(state_next)
                td_error = GAMMA*value_next+reward - value_this
                memory.push(env.make_folded_state(obs_queue), action, reward, done, td_error) #how to encoding TD-error?
            else:
                memory.push(env.make_folded_state(obs_queue), action, reward, done)

            if step % POLICY_UPDATE == 0 and training:
                agent.learn(memory, BATCH_SIZE)
                if step > STABLE_STEPS and stable:
                    agent.stable_learn(env.make_folded_state(obs_queue), action, reward, done)

            if step % TARGET_UPDATE == 0:
                agent.sync()

            if step % EVALUATE_FREQ == 0:
                avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
                with open("rewards.txt", "a") as fp:
                    fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
                if RENDER:
                    prefix = f"eval_{step//EVALUATE_FREQ:03d}"
                    os.mkdir(prefix)
                    for ind, frame in enumerate(frames):
                        with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                            frame.save(fp, format="png")
                #agent.save(os.path.join(SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
                done = True
                avg_reward_arr.append(avg_reward)
                progressive.set_description('current eval Avg Reward:{}'.format(avg_reward))

        plt.plot(range(len(avg_reward_arr)), avg_reward_arr, co[version], label=version)
        np_arr = np.array(avg_reward_arr)
        np.savetxt('./natural_result.txt'.format(version), np_arr, fmt='%f', delimiter=',')
        
        agent.save(os.path.join(SAVE_PREFIX, "model_{}.pth".format(version)))
        if COUNT_SHADE:
            action_queues.append(action_queue)
    if COUNT_SHADE:
        count_shade(action_queues, versions)
    plt.legend()
    plt.savefig('./result_running3.png')
