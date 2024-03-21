import numpy as np
import gym
from gym import spaces
import tetris_engine as game

SCREEN_WIDTH, SCREEN_HEIGHT = 400,200


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # open up a game state to communicate with emulator
        self.game_state = game.GameState()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(23,), dtype=np.float32)
        self.viewer = None


    def step(self, a):
        self._action_set = np.zeros([len(self._action_set)])
        self._action_set[a] = 1
        
        def board_data_tensor():
            board_array = np.array(self.game_state.board)
            converted_array = np.where(board_array == '.', 0, 1)
            return np.rot90(converted_array,k=3)
     
        def calculate_holes_vector():
            """Calculate the Holes vector for the given Tetris game board."""
            game_board = board_data_tensor()
            holes_vector = np.zeros(game_board.shape[1], dtype=int)
            for col in range(game_board.shape[1]):
                column_data = game_board[:, col]
                first_block_index = np.where(column_data == 1)[0]
                if len(first_block_index) > 0:
                    # Count holes by looking for zeros after the first block
                    holes_vector[col] = len(np.where(column_data[first_block_index[0]:] == 0)[0])
            return holes_vector

        def calculate_boundaries_vector():
            """Calculate the Boundaries vector for the given Tetris game board."""
            game_board = board_data_tensor()
            boundaries_vector = np.zeros(game_board.shape[1], dtype=int)
            for col in range(game_board.shape[1]):
                column_data = game_board[:, col]
                first_block_index = np.where(column_data == 1)[0]
                if len(first_block_index) > 0:
                    # Height is total rows - first block index
                    boundaries_vector[col] = game_board.shape[0] - first_block_index[0]
            return boundaries_vector     
        def piece_data_tensor():
            piece_map ={"I":0, "J": 1, "L":2, "O":3, "S":4, "T":5, "Z":6}
            piece_array = np.zeros(2)
            if self.game_state.fallingPiece:
                piece = self.game_state.fallingPiece
                piece_array[0] = piece_map[piece['shape']]
                piece_array[1] = piece['rotation']
            return piece_array
        

        reward = 0.0
        
        state, reward, terminal = self.game_state.frame_step(self._action_set)
        
        cleared_lines = self.game_state.lines
        simp_state = np.concatenate((calculate_holes_vector(),calculate_boundaries_vector(),piece_data_tensor(), np.array([cleared_lines])))
        
        return simp_state, reward, terminal, {}

    def get_image(self):
        return self.game_state.getImage()

    @property
    def n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        do_nothing = np.zeros(len(self._action_set))
        do_nothing[0] = 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(23,), dtype=np.float32)
        state, _, _= self.game_state.frame_step(do_nothing)
        return np.zeros(23)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

