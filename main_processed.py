import torch
import random, datetime, os
from agent import *
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
from model import linear_DQN
from torch import optim
import math
import numpy as np


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 1e-3
LR = 1e-4
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_done = 0

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = state.shape

policy_net = linear_DQN(22, n_actions).to(device)
target_net = linear_DQN(22, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = torch.argmax(policy_net(state))
            return action
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

piece_dict = {'Tu': 0, 'Tr': 1, 'Td': 2, 'Tl': 3, 'Jl': 4, 'Ju': 5, 'Jr': 6, 'Jd': 7, 'Zh': 8, 'Zv': 9, 'O': 10, 'Sh': 11, 'Sv': 12, 'Lr': 13, 'Ld': 14, 'Ll': 15, 'Lu': 16, 'Iv': 17, 'Ih': 18}
    
    
def process_board(board):
    # Returns board, holes and boundaries
    board = np.where(board == 239,0,1)
    holes = np.zeros(board.shape[1], dtype=int)
    boundaries = np.zeros(board.shape[1], dtype=int)

    for col in range(board.shape[1]):
        column_data = board[:, col]
        first_block_idx = np.where(column_data == 1)[0]
        if first_block_idx.size > 0:
            first_block_idx = first_block_idx[0]
            boundaries[col] = board.shape[0] - first_block_idx
            holes[col] = np.count_nonzero(column_data[first_block_idx:] == 0)
    
    return board,holes,boundaries

def concat_data(holes,boundaries,cur_piece,cleared_lines):
    combined_array = np.concatenate((holes, boundaries))
    final_array = np.append(combined_array, [cur_piece, cleared_lines])
    return final_array

    

    
def train():
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50
    
    best_reward = -float('inf')

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        env.reset()
        state = torch.zeros(22, dtype=torch.float32)
        total_reward = 0
        done = False

        # print the episode number
        print(f"Episode {i_episode}")

        while not done:
            # env.render()
            # Select and perform an action
            action = select_action(state)
            next_state, reward, done, info = env.step(action.item())
            
            cur_piece = piece_dict[info["current_piece"]]
            cleared_lines = info["number_of_lines"]
            board, holes, boundaries = process_board(env.board)
            state_data = concat_data(holes,boundaries,cur_piece,cleared_lines)
            state_data = state_data.copy()
            state_data = torch.tensor(state_data, dtype=torch.float32)
            reward = torch.tensor([reward], device=device)

            # Move to the next state
            state = state_data

            # Perform one step of the optimization (on the target network)
            optimize_model()

            total_reward += reward.item()

        # Print results
        print(f"Episode {i_episode} finished")
        print(f"Total reward: {total_reward}")

        # Save checkpoint if best reward
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'episode': i_episode,
                'model_state_dict': policy_net.state_dict(),  # Assuming policy_net is your model
                'optimizer_state_dict': optimizer.state_dict(),  # Assuming optimizer is defined outside
                'reward': total_reward,
            }, "model_checkpoint.pth")


        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)







def evaluate():
    ...

if __name__ == "__main__":
    train()