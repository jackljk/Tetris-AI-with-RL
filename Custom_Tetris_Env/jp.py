"""An environment wrapper to convert binary to discrete action space."""
import gym
from gym import Env
from gym import Wrapper
import numpy as np
from gym.spaces import Box

class JoypadSpace(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env, actions: list):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = ' '.join(button_list)

        self.observation_space = Box(low=0, high=np.inf, shape=(40,), dtype=np.float32)

    def step(self, action):
        """
        Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information

        """
        # take the step and record the output

        # Custom STEP:
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
            combined_array = np.concatenate((holes, boundaries,cur_piece))
            final_array = np.append(combined_array, [cleared_lines])
            return final_array

        action = int(action)
        state, reward, done, info = self.env.step(self._action_map[action])

        # Catch exception if current_piece is none which can happen at the start of the game
        try:
            cur_piece = piece_dict[info["current_piece"]]
        except:
            cur_piece = 0


        cur_piece_ohe = np.zeros(19)
        cur_piece_ohe[cur_piece] = 1

        cleared_lines = info["number_of_lines"]
        board, holes, boundaries = process_board(self.env.board)
        state_data = concat_data(holes,boundaries,cur_piece_ohe,cleared_lines)
        state_data = state_data.copy()
        
        state = state_data

        return state, reward, done, info

    def reset(self):
        """Reset the environment and return the initial observation."""
        self.env.reset()
        return np.zeros(40)
        return self.env.reset()

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        # get the old mapping of keys to actions
        old_keys_to_action = self.env.unwrapped.get_keys_to_action()
        # invert the keys to action mapping to lookup key combos by action
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        # create a new mapping of keys to actions
        keys_to_action = {}
        # iterate over the actions and their byte values in this mapper
        for action, byte in self._action_map.items():
            # get the keys to press for the action
            keys = action_to_keys[byte]
            # set the keys value in the dictionary to the current discrete act
            keys_to_action[keys] = action

        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]


# explicitly define the outward facing API of this module
__all__ = [JoypadSpace.__name__]