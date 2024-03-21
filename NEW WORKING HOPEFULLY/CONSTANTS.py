ACTION_MAP = [

    # O Actions
    [1, 1, 1, 1, 4],               # Move 'O' to x1
    [1, 1, 1, 4],                  # Move 'O' to x2
    [1, 1, 4],                     # Move 'O' to x3
    [1, 4],                        # Move 'O' to x4
    [0, 4],                        # Move 'O' to x5
    [3, 4],                        # Move 'O' to x6
    [3, 3, 4],                     # Move 'O' to x7
    [3, 3, 3, 4],                  # Move 'O' to x8
    [3, 3, 3, 3, 4],               # Move 'O' to x9

    # O PADDING
    [4],                           # Move 'O' to x10

    # z Actions
    [1, 1, 1, 1, 4],               # Move 'z' to x1
    [1, 1, 1, 4],                  # Move 'z' to x2
    [1, 1, 4],                     # Move 'z' to x3
    [1, 4],                        # Move 'z' to x4
    [0, 4],                        # Move 'z' to x5
    [3, 4],                        # Move 'z' to x6
    [3, 3, 4],                     # Move 'z' to x7
    [3, 3, 3, 4],                  # Move 'z' to x8 one less as piece is 3 long
    [2, 1, 1, 1, 1, 4],            # Move 'z' to x1 with 1 rotation
    [2, 1, 1, 1, 4],               # Move 'z' to x2 with 1 rotation
    [2, 1, 1, 4],                  # Move 'z' to x3 with 1 rotation
    [2, 1, 4],                     # Move 'z' to x4 with 1 rotation
    [2, 0, 4],                     # Move 'z' to x5 with 1 rotation
    [2, 3, 4],                     # Move 'z' to x6 with 1 rotation
    [2, 3, 3, 4],                  # Move 'z' to x7 with 1 rotation
    [2, 3, 3, 3, 4],               # Move 'z' to x8 with 1 rotation
    [2, 3, 3, 3, 3, 4],            # Move 'z' to x9 with 1 rotation

    # z Padding
    [4],                           # Move 'z' to x9
    [4],                           # Move 'z' to x10
    [4],                           # Move 'z' to x10

    # s Actions
    
    [1, 1, 1, 4],                  # Move 's' to x2 one less as piece is 3 long
    [1, 1, 4],                     # Move 's' to x3 
    [1, 4],                        # Move 's' to x4 
    [0, 4],                        # Move 's' to x5 
    [3, 4],                        # Move 's' to x6 
    [3, 3, 4],                     # Move 's' to x7 
    [3, 3, 3, 4],                  # Move 's' to x8
    [3, 3, 3, 3, 4],               # Move 's' to x9
    [2, 1, 1, 1, 1, 4],            # Move 's' to x1 with 1 rotation
    [2, 1, 1, 1, 4],               # Move 's' to x2 with 1 rotation
    [2, 1, 1, 4],                  # Move 's' to x3 with 1 rotation
    [2, 1, 4],                     # Move 's' to x4 with 1 rotation
    [2, 0, 4],                     # Move 's' to x5 with 1 rotation
    [2, 3, 4],                     # Move 's' to x6 with 1 rotation
    [2, 3, 3, 4],                  # Move 's' to x7 with 1 rotation
    [2, 3, 3, 3, 4],               # Move 's' to x8 with 1 rotation
    [2, 3, 3, 3, 3, 4],            # Move 's' to x9 with 1 rotation

    # s Padding 
    [4],                           # Move 's' to x1
    [4],                           # Move 's' to x10
    [4],                           # Move 's' to x10

    # I Actions
    [1, 1, 1, 1, 1, 4],            # Move 'I' to x1
    [1, 1, 1, 1, 4],               # Move 'I' to x2
    [1, 1, 1, 4],                  # Move 'I' to x3
    [1, 1, 4],                     # Move 'I' to x4
    [1, 4],                        # Move 'I' to x5
    [0, 4],                        # Move 'I' to x6
    [3, 4],                        # Move 'I' to x7
    [3, 3, 4],                     # Move 'I' to x8
    [3, 3, 3, 4],                  # Move 'I' to x9
    [3, 3, 3, 3, 4],               # Move 'I' to x10
    [2, 1, 1, 1, 4],               # Move 'I' to x1 with 1 rotation
    [2, 1, 1, 4],                  # Move 'I' to x2 with 1 rotation
    [2, 1, 4],                     # Move 'I' to x3 with 1 rotation
    [2, 0, 4],                     # Move 'I' to x4 with 1 rotation
    [2, 3, 4],                     # Move 'I' to x5 with 1 rotation
    [2, 3, 3, 4],                  # Move 'I' to x6 with 1 rotation
    [2, 3, 3, 3, 4],               # Move 'I' to x7 with 1 rotation no more as piece is 4 long

    # I Padding 
    [4],                           # Move 'I' to x8
    [4],                           # Move 'I' to x9
    [4],                           # Move 'I' to x10

    # J Actions
    [1, 1, 1, 1, 4],               # Move 'J' to x1
    [1, 1, 1, 4],                  # Move 'J' to x2
    [1, 1, 4],                     # Move 'J' to x3
    [1, 4],                        # Move 'J' to x4
    [0, 4],                        # Move 'J' to x5
    [3, 4],                        # Move 'J' to x6
    [3, 3, 4],                     # Move 'J' to x7
    [3, 3, 3, 4],                  # Move 'J' to x8 no more as piece is 3 long
    [2, 1, 1, 1, 1, 1, 4],         # Move 'J' to x1 with 1 rotation
    [2, 1, 1, 1, 1, 4],            # Move 'J' to x2 with 1 rotation
    [2, 1, 1, 1, 4],               # Move 'J' to x3 with 1 rotation
    [2, 1, 1, 4],                  # Move 'J' to x4 with 1 rotation
    [2, 1, 4],                     # Move 'J' to x5 with 1 rotation
    [2, 0, 4],                     # Move 'J' to x6 with 1 rotation
    [2, 3, 4],                     # Move 'J' to x7 with 1 rotation
    [2, 3, 3, 4],                  # Move 'J' to x8 with 1 rotation
    [2, 3, 3, 3, 4],               # Move 'J' to x9 with 1 rotation
    [2, 2, 1, 1, 1, 1, 4],         # Move 'J' to x1 with 2 rotation
    [2, 2, 1, 1, 1, 4],            # Move 'J' to x2 with 2 rotation
    [2, 2, 1, 1, 4],               # Move 'J' to x3 with 2 rotation
    [2, 2, 1, 4],                  # Move 'J' to x4 with 2 rotation
    [2, 2, 4],                     # Move 'J' to x5 with 2 rotation
    [2, 2, 3, 4],                  # Move 'J' to x6 with 2 rotation
    [2, 2, 3, 3, 4],               # Move 'J' to x7 with 2 rotation
    [2, 2, 3, 3, 3, 4],            # Move 'J' to x8 with 2 rotation
    [5, 1, 1, 1, 1, 4],            # Move 'J' to x1 with 3 rotation
    [5, 1, 1, 1, 4],               # Move 'J' to x2 with 3 rotation
    [5, 1, 1, 4],                  # Move 'J' to x3 with 3 rotation
    [5, 1, 4],                     # Move 'J' to x4 with 3 rotation
    [5, 0, 4],                     # Move 'J' to x5 with 3 rotation
    [5, 3, 4],                     # Move 'J' to x6 with 3 rotation
    [5, 3, 3, 4],                  # Move 'J' to x7 with 3 rotation
    [5, 3, 3, 3, 4],               # Move 'J' to x8 with 3 rotation
    [5, 3, 3, 3, 3, 4],            # Move 'J' to x9 with 3 rotation

    # J Padding
    [4],                           # Move 'J' to x9
    [4],                           # Move 'J' to x10
    [4],                           # Move 'J' to x10
    [4],                           # Move 'J' to x9
    [4],                           # Move 'J' to x10
    [4],                           # Move 'J' to x10

    # L Actions
    [1, 1, 1, 1, 4],               # Move 'L' to x1
    [1, 1, 1, 4],                  # Move 'L' to x2
    [1, 1, 4],                     # Move 'L' to x3
    [1, 4],                        # Move 'L' to x4
    [0, 4],                        # Move 'L' to x5
    [3, 4],                        # Move 'L' to x6
    [3, 3, 4],                     # Move 'L' to x7
    [3, 3, 3, 4],                  # Move 'L' to x8 no more as piece is 3 long
    [2, 1, 1, 1, 1, 1, 4],         # Move 'L' to x1 with 1 rotation
    [2, 1, 1, 1, 1, 4],            # Move 'L' to x2 with 1 rotation
    [2, 1, 1, 1, 4],               # Move 'L' to x3 with 1 rotation
    [2, 1, 1, 4],                  # Move 'L' to x4 with 1 rotation
    [2, 1, 4],                     # Move 'L' to x5 with 1 rotation
    [2, 0, 4],                     # Move 'L' to x6 with 1 rotation
    [2, 3, 4],                     # Move 'L' to x7 with 1 rotation
    [2, 3, 3, 4],                  # Move 'L' to x8 with 1 rotation
    [2, 3, 3, 3, 4],               # Move 'L' to x9 with 1 rotation
    [2, 2, 1, 1, 1, 1, 4],         # Move 'L' to x1 with 2 rotation
    [2, 2, 1, 1, 1, 4],            # Move 'L' to x2 with 2 rotation
    [2, 2, 1, 1, 4],               # Move 'L' to x3 with 2 rotation
    [2, 2, 1, 4],                  # Move 'L' to x4 with 2 rotation
    [2, 2, 4],                     # Move 'L' to x5 with 2 rotation
    [2, 2, 3, 4],                  # Move 'L' to x6 with 2 rotation
    [2, 2, 3, 3, 4],               # Move 'L' to x7 with 2 rotation
    [2, 2, 3, 3, 3, 4],            # Move 'L' to x8 with 2 rotation no more as piece is 3 long
    [5, 1, 1, 1, 1, 4],            # Move 'L' to x1 with 3 rotation
    [5, 1, 1, 1, 4],               # Move 'L' to x2 with 3 rotation
    [5, 1, 1, 4],                  # Move 'L' to x3 with 3 rotation
    [5, 1, 4],                     # Move 'L' to x4 with 3 rotation
    [5, 0, 4],                     # Move 'L' to x5 with 3 rotation
    [5, 3, 4],                     # Move 'L' to x6 with 3 rotation
    [5, 3, 3, 4],                  # Move 'L' to x7 with 3 rotation
    [5, 3, 3, 3, 4],               # Move 'L' to x8 with 3 rotation
    [5, 3, 3, 3, 3, 4],            # Move 'L' to x9 with 3 rotation

    # L Padding
    [4],                           # Move 'L' to x9
    [4],                           # Move 'L' to x10
    [4],                           # Move 'L' to x10
    [4],                           # Move 'L' to x9
    [4],                           # Move 'L' to x10
    [4],                           # Move 'L' to x10

    # T Actions
    [1, 1, 1, 1, 4],               # Move 'T' to x1
    [1, 1, 1, 4],                  # Move 'T' to x2
    [1, 1, 4],                     # Move 'T' to x3
    [1, 4],                        # Move 'T' to x4
    [0, 4],                        # Move 'T' to x5
    [3, 4],                        # Move 'T' to x6
    [3, 3, 4],                     # Move 'T' to x7
    [3, 3, 3, 4],                  # Move 'T' to x8 no more as piece is 3 long
    [2, 1, 1, 1, 1, 1, 4],         # Move 'T' to x1 with 1 rotation
    [2, 1, 1, 1, 1, 4],            # Move 'T' to x2 with 1 rotation
    [2, 1, 1, 1, 4],               # Move 'T' to x3 with 1 rotation
    [2, 1, 1, 4],                  # Move 'T' to x4 with 1 rotation
    [2, 1, 4],                     # Move 'T' to x5 with 1 rotation
    [2, 0, 4],                     # Move 'T' to x6 with 1 rotation
    [2, 3, 4],                     # Move 'T' to x7 with 1 rotation
    [2, 3, 3, 4],                  # Move 'T' to x8 with 1 rotation
    [2, 3, 3, 3, 4],               # Move 'T' to x9 with 1 rotation
    [2, 2, 1, 1, 1, 1, 4],         # Move 'T' to x1 with 2 rotation
    [2, 2, 1, 1, 1, 4],            # Move 'T' to x2 with 2 rotation
    [2, 2, 1, 1, 4],               # Move 'T' to x3 with 2 rotation
    [2, 2, 1, 4],                  # Move 'T' to x4 with 2 rotation
    [2, 2, 4],                     # Move 'T' to x5 with 2 rotation
    [2, 2, 3, 4],                  # Move 'T' to x6 with 2 rotation
    [2, 2, 3, 3, 4],               # Move 'T' to x7 with 2 rotation
    [2, 2, 3, 3, 3, 4],            # Move 'T' to x8 with 2 rotation no more as piece is 3 long
    [5, 1, 1, 1, 1, 4],            # Move 'T' to x1 with 3 rotation
    [5, 1, 1, 1, 4],               # Move 'T' to x2 with 3 rotation
    [5, 1, 1, 4],                  # Move 'T' to x3 with 3 rotation
    [5, 1, 4],                     # Move 'T' to x4 with 3 rotation
    [5, 0, 4],                     # Move 'T' to x5 with 3 rotation
    [5, 3, 4],                     # Move 'T' to x6 with 3 rotation
    [5, 3, 3, 4],                  # Move 'T' to x7 with 3 rotation
    [5, 3, 3, 3, 4],               # Move 'T' to x8 with 3 rotation
    [5, 3, 3, 3, 3, 4],            # Move 'T' to x9 with 3 rotation

    # T Padding
    [4],                           # Move 'L' to x9
    [4],                           # Move 'L' to x10
    [4],                           # Move 'L' to x10
    [4],                           # Move 'L' to x9
    [4],                           # Move 'L' to x10
    [4],                           # Move 'L' to x10
]