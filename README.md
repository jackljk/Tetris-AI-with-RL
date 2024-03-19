# Reinforcement Learning Tetris AI

## Description
Welcome to our project, Reinforcement Learning Tetris AI. This project attempts to train a reinforcement learning agent to play Tetris efficiently. 



## Table of Contents

1. [Installation](#installation)
2. [Instructions on How to Run the Code](#instructions)


### Installation
**Disclaimer: The instructions below are designed for Mac devices**
1. Run this install in your terminal `pip install gym-tetris`
2. Go to your terminal to find where gym-tetris got installed. You can do this by running `python -v`.
3. Run `import gym-tetris` then run `gym-tetris.__file__` to get the file path
4. Open a new terminal then cd into this file path
5. Type `vi tetris_env.py`. This will allow you to edit the file in the terminal
6. Press i on your keyboard and you can start editing the file. Add the function 
```
@property
def board(self):
    """Return the Tetris board from NES RAM."""
    return self.ram[0x0400:0x04C8].reshape((20, 10)).copy()
```
7. Press esc on your keyboard then scroll to the end of the file to type :wq to save the code.
8. You will be able to access the `board` variable in your code from now on
