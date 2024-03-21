from customDQN import DQNAgent
from tetris_env_2 import TetrisEnv
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm
from CONSTANTS import PIECE_TO_ACTION
        

# Run dqn with Tetris
def dqn():
    env = TetrisEnv()
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

    agent = DQNAgent(env.n_actions,
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
            curr_piece = env.game_state.fallingPiece['shape']
            states = env.get_all_states()
            best_state = agent.best_state(states) # get the best state from the model
            
            best_action = None
            for i, state in enumerate(states): # get the action that corresponds to the best state
                if state == best_state:
                    best_action = i + PIECE_TO_ACTION[curr_piece]
                    break
            
            reward, done = env.step(best_action) # take the best action
            

            agent.add_to_memory(current_state, best_state, reward, done) # add the play to the replay memory buffer

            # Save to logs
            current_state = best_state.copy()
            steps += 1

        scores.append(env.game_state.get_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            print(f'Episode: {episode}, Score: {avg_score}, Min: {min_score}, Max: {max_score}, Epsilon: {agent.epsilon}')


