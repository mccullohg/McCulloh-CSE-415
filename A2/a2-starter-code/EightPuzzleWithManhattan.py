""" EightPuzzleWithManhattan.py

This is a formulation of the Eight Puzzle using a Manhattan distance heuristic.
Partnership? (YES or NO): YES
Student Name 1: Gordon McCulloh
Student Name 2: Arnav Khera

UW NetID: 2027940
CSE 415, Spring 2021, University of Washington

The Manhattan distance counts the horizontal and vertical displacement of
each tile from its position in the goal state, then sums this value.

Usage:
python3 EightPuzzleWithManhattan '[[3,0,1],[6,4,2],[7,8,5]]'
"""

# Import EightPuzzle code for the puzzle structure
from EightPuzzle import *

def h(s):
    # Heuristic function - Manhattan distance
    idx = 0
    for row in range(3):
        for col in range(3):
            # Correct position of the tile
            (goal_row, goal_col) = (s.b[row][col] // 3, s.b[row][col] % 3)
            # Sum the Manhattan distance
            if s.b[row][col] != 0:
                idx = idx + abs(row-goal_row) + abs(col-goal_col)
    return idx
