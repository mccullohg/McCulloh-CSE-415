'''
Names: Gordon McCulloh, Arnav Khera
UWNet IDs: mcculloh, akhera29
'''

from game_engine import genmoves

class BackgammonPlayer:
    def __init__(self):
        self.GenMoveInstance = genmoves.GenMoves()
        self.states = 0  # initialize states explored
        self.prune = True  # declare alpha-beta pruning or minimax
        self.cutoff = 0  # initialize states cut off
        self.maxply = 0  # initialize maxply
        self.alpha = -10000  # initialize alpha
        self.beta = 10000  # initialize beta
        self.best = {}  # initialize dictionary of best moves


    # return a string representation of UWNet IDs (Gordon, Arnav)
    def nickname(self):
        return "mcculloh akhera29"

    # If prune==True, then the Move method should use Alpha-Beta Pruning
    # otherwise Minimax
    def useAlphaBetaPruning(self, prune=False):
        self.states = 0
        self.cutoff = 0
        self.prune = prune

    # Returns a tuple containing the number explored
    # states as well as the number of cutoffs.
    def statesAndCutoffsCounts(self):
        return (self.states, self.cutoff)

    # Given a ply, it sets a maximum for how far an agent
    # should go down in the search tree. maxply=2 indicates that
    # our search level will go two level deep.
    def setMaxPly(self, maxply=2):
        self.maxply = maxply

    # If not None, it updates the internal static evaluation
    # function to be func
    def useSpecialStaticEval(self, func):
        if func is not None:
            self.staticEval = func

    # Given a state and a roll of dice, it returns the best move for
    # the state.whose_move.
    # Keep in mind: a player can only pass if the player cannot move any checker with that roll
    def move(self, state, die1=1, die2=6):
        self.best.clear()  # clear dictionary memory
        value = self.MinimaxSearch(state, die1, die2, state.whose_move, self.maxply, self.alpha, self.beta)
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
                    H -= (24-ii)  # index white checkers CCW from 24
                    stack += 1
                else:
                    H += (ii+1)  # index red checkers CW from 1
                    stack += 1
                # Potential bearing off checkers - weighted +- 3
                # Single checkers are susceptible to bearing off
                if stack == 1:
                    if checker == 0:
                        red_adversary = 0  # initialize the number of red checkers in range
                        for jj in range(ii+1, min(ii+13, 24)):  # within two die rolls
                            if checker == 1:
                                red_adversary += 1
                        H -= 3*red_adversary  # count red bearing off checkers against white
                    else:
                        white_adversary = 0  # initialize the number of white checkers in range
                        for jj in range(max(ii-12,0), ii):  # within two die rolls
                            if checker == 0:
                                white_adversary += 1
                        H += 3*white_adversary  # count white bearing off checkers against red

        # Checkers on the bar - weighted +- 25
        # Checkers on the bar are undesirable
        for checker in state.bar:
            if checker == 0:
                H -= 25
            else:
                H += 25

        # Checkers removed - weighted +- 50
        # Checkers removed indicate that the player is closer to winning
        H += 50*len(state.white_off)
        H -= 50*len(state.red_off)

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
    def MinimaxSearch(self, state, die1, die2, whose_move, maxply, alpha, beta):
        # Search root node
        if maxply == 0:
            return self.staticEval(state)

        # Initialize best value
        if whose_move == 0:  # MIN
            value = self.alpha
        else:  # MAX
            value = self.beta

        # Initialize best move and number of states kept
        move = None
        keep = 0

        # Get the list of all legal moves from function legalBG
        legal = self.legalBG(state, die1, die2)

        # Minimax search algorithm
        for (M, S) in legal:
            self.states += 1  # count number of states expanded
            keep += 1  # count number of states kept
            # Examine branch node
            new_value = self.MinimaxSearch(S, die1, die2, S.whose_move, maxply-1, alpha, beta)
            if (whose_move==0 and new_value>value) or (whose_move==1 and new_value<value):
                value = new_value  # overwrite new max/min
                move = M  # store best move

            # Alpha-beta pruning sub-algorithm
            if self.prune:
                if whose_move == 0:  # White
                    alpha = max(alpha, value)
                else:  # Red
                    beta = min(beta, value)

                if alpha >= beta:
                    self.cutoff = self.cutoff + len(legal) - keep  # update states cut off
                    break

        self.best[value] = move  # store best move and value in dictionary
        return value
