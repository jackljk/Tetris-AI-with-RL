from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
        

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    # log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0
        best_action = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            states, reward, done, _ = env.step(best_action) # take a step to start the round
            best_state = agent.best_state(states) # get the best state from the model
            
            best_action = None
            for action, state in states.items(): # get the action that corresponds to the best state
                if state == best_state:
                    best_action = action
                    break
            
            # execute the best action and get the reward
            for act in best_action:
                _, _reward, done, _ = env.game_state.frame_step(env._action_set[act])
                reward += _reward
                if done:
                    break
                
            reward = reward/len(best_action) # normalize the reward
            
            agent.add_to_memory(current_state, states[best_action], reward, done) # add the play to the replay memory buffer

            # Save to logs
            current_state = states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            print(f'Episode: {episode}, Score: {avg_score}, Min: {min_score}, Max: {max_score}, Epsilon: {agent.epsilon}')


