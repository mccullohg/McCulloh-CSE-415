'''
Name(s):
UW netid(s):
'''

from game_engine import genmoves

class BackgammonPlayer:
    def __init__(self):
        self.GenMoveInstance = genmoves.GenMoves()
        # feel free to create more instance variables as needed.
        self.maxply = 0  # initialize maxply
        self.best = {}  # initialize dictionary of best moves

    # TODO: return a string containing your UW NETID(s)
    # For students in partnership: UWNETID + " " + UWNETID
    def nickname(self):
        # TODO: return a string representation of your UW netid(s)
        return "akhera29 mcculloh"

    # Given a ply, it sets a maximum for how far an agent
    # should go down in the search tree. Count the chance nodes
    # as a ply too!
    def setMaxPly(self, maxply=2):
        # TODO: set the max ply
        self.maxply = maxply

    # If not None, it update the internal static evaluation
    # function to be func
    def useSpecialStaticEval(self, func):
        # TODO: update your staticEval function appropriately
        if func is not None:
            self.staticEval = func

    # Given a state and a roll of dice, it returns the best move for
    # the state.whose_move
    # Keep in mind: a player can only pass if the player cannot move any checker with that role
    def move(self, state, die1, die2):
        # TODO: return a move for the current state and for the current player.
        self.best.clear()  # clear dictionary memory
        value = self.ExpectiMinimaxSearch(state, die1, die2, state.whose_move, self.maxply)
        return self.best[value]  # return the move (key) for the best calculated value from minimax

    # Calculated static evaluation function from available information in boardState.py
    # White (W) wants higher function value - MAX
    # Red (R) wants lower function value - MIN
    def staticEval(self, state):
        # Initialize static evaluation
        H = 0

        # Examine each checker on the board
        for ii in range(24):  # search all positions
            stack = 0  # initialize the number of checkers on a position
            for checker in state.pointLists[ii]:
                # Purely counting position away from goal - weighted +- 1
                if checker == 0:
                    H -= (24 - ii)  # index white checkers CCW from 24
                    stack += 1
                else:
                    H += (ii + 1)  # index red checkers CW from 1
                    stack += 1
                # Potential bearing off checkers - weighted +- 3
                # Single checkers are susceptible to bearing off
                if stack == 1:
                    if checker == 0:
                        red_adversary = 0  # initialize the number of red checkers in range
                        for jj in range(ii + 1, min(ii + 13, 24)):  # within two die rolls
                            if checker == 1:
                                red_adversary += 1
                        H -= 3 * red_adversary  # count red bearing off checkers against white
                    else:
                        white_adversary = 0  # initialize the number of white checkers in range
                        for jj in range(max(ii - 12, 0), ii):  # within two die rolls
                            if checker == 0:
                                white_adversary += 1
                        H += 3 * white_adversary  # count white bearing off checkers against red

        # Checkers on the bar - weighted +- 25
        # Checkers on the bar are undesirable
        for checker in state.bar:
            if checker == 0:
                H -= 25
            else:
                H += 25

        # Checkers removed - weighted +- 50
        # Checkers removed indicate that the player is closer to winning
        H += 50 * len(state.white_off)
        H -= 50 * len(state.red_off)

        # white wants positive, red wants negative
        return H

    # List all of the legal moves according to the rules of Backgammon for a given state
    def legalBG(self, state, die1, die2):
        moves = self.GenMoveInstance.gen_moves(state, state.whose_move, die1, die2)  # generate all moves
        legal = []  # initialize legal move list
        find_moves = True  # initialize search operator
        legal_moves = False  # initialize a pass state
        # Detect a winning state, do not check multiple pass states
        while find_moves:
            try:
                move = next(moves)  # call a (move, state) tuple from list of legal moves
                if move[0] != 'p':  # winning state, add move
                    legal.append(move)
                    legal_moves = True
            except StopIteration as E:  # built-in exception to while loop
                find_moves = False
        if not legal_moves:  # pass state
            legal.append(('p', state))  # state does not change, add pass
        return legal

    # Execute minimax search or alpha-beta minimax search
    def ExpectiMinimaxSearch(self, state, die1, die2, whose_move, maxply):
        # Search root node
        if maxply == 0:
            return self.staticEval(state)

        # Initialize best value
        if whose_move == 0:  # MIN
            value = -1000000
        elif whose_move == 1:  # MAX
            value = 1000000
        else:
            value = 0

        # Initialize best move and number of states kept
        move = None

        # Get the list of all legal moves from function legalBG
        legal = self.legalBG(state, die1, die2)

        # Minimax search algorithm
        for (M, S) in legal:
            # Examine branch node
            if whose_move >= 0:
                new_value = self.ExpectiMinimaxSearch(S, die1, die2, S.whose_move - 2, maxply - 1)
                if (whose_move == 0 and new_value > value) or (whose_move == 1 and new_value < value):
                    value = new_value  # overwrite new max/min
                    move = M  # store best move
            else:
                value += 1/36 * self.ExpectiMinimaxSearch(S, die1, die2, S.whose_move + 2, maxply - 1)

        self.best[value] = move  # store best move and value in dictionary
        return value
