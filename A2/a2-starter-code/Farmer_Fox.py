'''Farmer_Fox.py
by Gordon McCulloh
UWNetID: mcculloh
Student number: 2027940

Assignment 2, Part 1, in CSE 415, Spring 2021.
 
This file contains my problem formulation for the problem of
the Farmer, Fox, Chicken, and Grain.
'''

# Put your formulation of the Farmer-Fox-Chicken-and-Grain problem here.
# Be sure your name, uwnetid, and 7-digit student number are given above in 
# the format shown.

#<METADATA>
SOLUTION_VERSION = "1.0"
PROBLEM_NAME = "The Farmer-Fox-Chicken-and-Grain Problem"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['Gordon McCulloh']
PROBLEM_CREATION_DATE = "14-APR-2021"

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''A farmer need to take a fox, chicken, and sack of grain across a river using a
 small boat. He can only take one of the three items in the boat with him at one time. 
 The fox must never be left alone with the chicken, and the chicken must never be left
 alone with the grain. How can he get everything across the river?
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

#<COMMON_CODE>
# Array indices for problem objects
Fa=0  # Farmer
F=1  # Fox
C=2  # Chicken
G=3  # Grain
LEFT=0  # left side of river
RIGHT=1  # right side of river

class State():

  def __init__(self, d=None):
    if d==None:
      d = {'agents':[0,0,0,0],
           'ferry':LEFT}
    self.d = d

  def __eq__(self,s2):
    for prop in ['agents', 'ferry']:
      if self.d[prop] != s2.d[prop]: return False
    return True

  def __str__(self):
    # Produces a textual description of a state.
    p = self.d['agents']
    txt = "\n Farmer is on the "+ self.__side__(p,Fa) +"\n"
    txt += " Fox is on the "+ self.__side__(p,F) +"\n"
    txt += " Chicken is on the " + self.__side__(p,C) + "\n"
    txt += " Grain is on the " + self.__side__(p,G) + "\n"
    side='left'
    if self.d['ferry']==1: side='right'
    txt += " Boat is on the "+side+".\n"
    return txt

  def __side__(self,p,jj):
    # return the side of the farmer, fox, chicken, or grain
    if p[jj] == 0:
      return 'left'
    else:
      return 'right'

  def __hash__(self):
    return (self.__str__()).__hash__()

  def copy(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State({})
    news.d['agents']=[jj for jj in self.d['agents']]
    news.d['ferry'] = self.d['ferry']
    return news

  def can_move(self,p1,p2):
    '''Tests whether it's legal to move the boat and take
     p1 and p2 (objects) with it.'''
    side = self.d['ferry'] # Where the boat is.
    p = self.d['agents']
    # Farmer (p1) and F/C/G (p2) must be on the same side as the boat
    if p[p1] != side or p[p2] != side: return False
    # Remaining F/C/G (p2)
    # Fox cannot be left with the chicken, chicken cannot be left with grain
    remaining = [ii for ii in [1,2,3] if p[ii] == side and ii != p2]
    if len(remaining) > 1:
      if remaining[0] == 1 and remaining[1] == 2: return False  # FC
      if remaining[0] == 2 and remaining[1] == 3: return False  # CG
      if remaining[0] == 1 and remaining[1] == 2 and remaining[2] == 3: return False  # FCG
    return True


  def move(self,p1,p2):
    '''Assuming it's legal to make the move, this computes
     the new state resulting from moving the boat carrying
     the Farmer (p1) and F/C/G (p2).'''
    news = self.copy()      # start with a deep copy.
    side = self.d['ferry']       # where is the boat?
    p = news.d['agents']          # get the array of arrays of agents.
    p[p1] = 1-side
    if p2 != 0:  # farmer is taking F,C, or G
      p[p2] = 1-side
    news.d['ferry'] = 1-side      # Move the boat itself.
    return news

def goal_test(s):
  '''If all objects are on the right, then s is a goal state.'''
  p = s.d['agents']
  for ii in p:
    if ii != 1: return False  # all objects not on RIGHT == 1
  return True

def goal_message(s):
  return "Congratulations on successfully guiding the farmer, fox, " \
         "chicken, and grain across the river!"

class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)
#</COMMON_CODE>

#<INITIAL_STATE>
CREATE_INITIAL_STATE = lambda : State(d={'agents':[0,0,0,0], 'ferry':LEFT })
#</INITIAL_STATE>

#<OPERATORS>
# Farmer crosses alone, w/ fox, w/ chicken, w/ grain.
obj_combinations = [(0,0),(0,1),(0,2),(0,3)]

def x_descrip(ii):
  if ii == 0:
    return 'Farmer crosses alone'
  if ii == 1:
    return 'Farmer crosses with fox'
  if ii == 2:
    return 'Farmer crosses with chicken'
  if ii == 3:
    return 'Farmer crosses with grain'

OPERATORS = [Operator(
  x_descrip(p2),
  lambda s, p1=p1, p2=p2: s.can_move(p1,p2),
  lambda s, p1=p1, p2=p2: s.move(p1,p2))
  for (p1,p2) in obj_combinations]
#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>
