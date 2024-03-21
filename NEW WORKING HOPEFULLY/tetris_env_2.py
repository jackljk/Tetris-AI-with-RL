import numpy as np
import gym
from gym import spaces
import tetris_engine as game
from CONSTANTS import *

SCREEN_WIDTH, SCREEN_HEIGHT = 400,200


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # open up a game state to communicate with emulator
        self.game_state = game.GameState()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(146,), dtype=np.float32)
        self.viewer = None


    def step(self, a):
        self._action_set = np.zeros([len(self._action_set)])
        self._action_set[a] = 1
        
        def board_data_tensor(board):
            board_array = np.array(board)
            converted_array = np.where(board_array == '.', 0, 1)
            return converted_array.T
        
        def calculate_possible_states(board, piece):
            """
            Calculate all possible states for a given Tetris piece placement on the board.
            
            :param board: A 2D list representing the Tetris board (0 = empty, 1 = filled).
            :param piece: A representation of the Tetris piece, including its shape and rotations.
            :return: A list of states, where each state is a dictionary with 'holes', 'bumpiness', and 'lines_cleared'.
            """
            states = [[0, 0, 0] for _ in range(48)]


            rotations = get_rotations(piece['shape'])
            for rotation in rotations:
                new_piece = piece.copy()
                new_piece['rotation'] = rotation

                # make a copy of the board
                sim_board = self.game_state.board.copy()
                # Simulate dropping the piece at x_position with the current rotation
                sim_boards = self.game_state.simulate_drop(sim_board, new_piece)
                holes, boundaries, lines_cleared_list = [], [], []
                for sim_board in sim_boards:
                    if sim_board == 0:
                        holes.append(0)
                        boundaries.append(0)
                        lines_cleared_list.append(0)
                        continue

                    sim_board = board_data_tensor(sim_board)
                    holes_vector, boundaries_vector = holes_boundary_vector(sim_board)
                    holes.append(np.sum(holes_vector))
                    boundaries.append(np.sum(boundaries_vector))
                    lines_cleared_list.append(lines_cleared(sim_board))
                    
                
                for i in range(len(holes)):
                    state = [
                        holes[i],
                        boundaries[i],
                        lines_cleared_list[i]
                    ]
                    states[i + rotation*12] = state
            
            return states
        
        def get_rotations(piece):
            """
            Get all possible rotations for a given Tetris piece.
            
            :param piece: A representation of the Tetris piece, including its shape and rotations.
            :return: A list of all possible rotations for the given piece.
            """
            if piece == 'O':
                return [0]
            if piece == 'I' or piece == 'Z' or piece == 'S':
                return [0, 1]
            return [0, 1, 2, 3]
        
        def lines_cleared(board):
            """Calculate the number of lines cleared for the given Tetris game board."""
            return len([i for i, row in enumerate(board) if 0 not in row])      
     
        def holes_boundary_vector(board):
            """Calculate the Holes vector for the given Tetris game board."""
            boundaries_vector = np.zeros(board.shape[1], dtype=int)
            holes_vector = np.zeros(board.shape[1], dtype=int)
            for col in range(board.shape[1]):
                column_data = board[:, col]
                first_block_index = np.where(column_data == 1)[0]
                if len(first_block_index) > 0:
                    # Count holes by looking for zeros after the first block
                    holes_vector[col] = len(np.where(column_data[first_block_index[0]:] == 0)[0])
                    # Height is total rows - first block index
                    boundaries_vector[col] = board.shape[0] - first_block_index[0]

            return holes_vector, boundaries_vector
   
        
        def piece_data_tensor():
            piece_map ={"I":0, "J": 1, "L":2, "O":3, "S":4, "T":5, "Z":6}
            piece_array = np.zeros(2)
            if self.game_state.fallingPiece:
                piece = self.game_state.fallingPiece
                piece_array[0] = piece_map[piece['shape']]
                piece_array[1] = piece['rotation']
            return piece_array

        curr_board = board_data_tensor(self.game_state.board)

        curr_piece = self.game_state.fallingPiece
        if not curr_piece:
            curr_piece = {
                'shape': 'O',
                'rotation': 0,
                'x': 0,
                'y': 0
            }
        states = calculate_possible_states(curr_board, curr_piece)

        reward = 0.0
        
        state, reward, terminal, end_round = self.game_state.frame_step(self._action_set)
        
    
        states = np.concatenate([states, piece_data_tensor()])
        return states, reward, terminal, {}

    def get_image(self):
        return self.game_state.getImage()

    @property
    def n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        do_nothing = np.zeros(len(self._action_set))
        do_nothing[0] = 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(146,), dtype=np.float32)
        state, _, _= self.game_state.frame_step(do_nothing)
        return np.zeros(146)

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

