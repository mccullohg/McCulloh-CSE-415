""" EightPuzzleWithHamming.py

This is a formulation of the Eight Puzzle using a hamming heuristic function.
Partnership? (YES or NO): YES
Student Name 1: Gordon McCulloh
Student Name 2: Arnav Khera

UW NetID: 2027940
CSE 415, Spring 2021, University of Washington

Hamming counts the number of tiles out of place in the eight puzzle,
but ignores the blank.

Usage:
python3 EightPuzzleWithHamming '[[3,0,1],[6,4,2],[7,8,5]]'
"""

# Import EightPuzzle code for the puzzle structure
from EightPuzzle import *

# Goal state
gamma = [[0, 1, 2], [3, 4, 5],[6, 7, 8]]

def h(s):
    # Heuristic function - return # of tiles out of place (except 0 tile)
    idx = 0
    for row in range(3):
        for col in range(3):
            # Count tile if not matching the goal state and not blank
            if s.b[row][col] != gamma[row][col] and s.b[row][col] != 0:
                idx += 1
    return idx
