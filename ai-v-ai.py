#!/usr/bin/python

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from collections import deque
import time
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

NUMBER_OF_CELLS=33
class Game(object):
  def __init__(self, p1, p2):
    self.p1turn=True
    self.squares = [0]*NUMBER_OF_CELLS #0 nothing 1 redPara 2 bluePara 3 redSoldier 4 blueSoldier 5 redTank 6 blueTank 7 redTrench 8 blueTrench 9 redSpy 10 blueSpy
    redsquares = [0, 1, 2, 30, 31, 32]
    bluesquares = [3, 9, 13, 19, 23, 29]
    for redSquare in redsquares:
      self.squares[redSquare] = 1
    for blueSquare in bluesquares:
      self.squares[blueSquare] = 2
    self.p1 = p1
    self.p2 = p2
    self.p1bases = 0
    self.p2bases = 0
    self.adjacency = adjacency = [\
                  [4],             \
                  [6],             \
                  [8],             \
                  [4],             \
                  [0, 3, 5, 10],   \
                  [4, 6],          \
                  [1, 5, 7, 11],   \
                  [6, 8],          \
                  [2, 7, 9, 12],   \
                  [8],             \
                  [4, 14],         \
                  [6, 16],         \
                  [8, 18],         \
                  [14],            \
                  [10, 13, 15, 20],\
                  [14, 16],        \
                  [11, 15, 17, 21],\
                  [16, 18],        \
                  [12, 17, 19, 22],\
                  [18],            \
                  [14, 24],        \
                  [16, 26],        \
                  [18, 28],        \
                  [24],            \
                  [20, 23, 25, 30],\
                  [24, 26],        \
                  [21, 25, 27, 31],\
                  [26, 28],        \
                  [22, 27, 29, 32],\
                  [28],            \
                  [24],            \
                  [26],            \
                  [28]]

  def getLegalMoves(self, playernum):
    parachutes = []
    bombs = []
    makeSoldiers = []
    moveSoldiers = []
    makeTanks = []
    moveTanks = []
    makeTrenches = []
    makeSpies = []
    moveSpies = []
    player = self.p1 if playernum == 1 else self.p2
    squares = self.squares
    for i,square in enumerate(squares):
      if square == 0: #square has nothing
        parachutes.append(i)
        if player.turnsToBomb == 0:
          bombs.append(i)
      elif [square,playernum] in [[1,1],[2,2]]: #square has this player's parachute
        makeSoldiers.append(i)
        makeTrenches.append(i)
      elif [square,playernum] in [[3,1],[4,2]]: #square has this player's soldier
        for adjacentI in self.adjacency[i]: #go through adjacenct indices
          if squares[adjacentI] not in [5,6,7,8]:
            moveSoldiers.append([i,adjacentI]) #add them if soldier can kill them
        makeTanks.append(i)
      elif ([square,playernum] in [[5,1],[6,2]]): #square has this player's tank
        for twoFarAdjacentI in self.getTwoFarAdjacencySet(i): #go through two far indices where tank can move
          moveTanks.append([i,twoFarAdjacentI])
      elif ([square,playernum] in [[7,1],[8,2]]): #square has this player's trench
        makeSpies.append(i)
      elif ([square,playernum] in [[9,1],[10,2]]): #square has this player's spy
        for adjacentI in self.adjacency[i]:
          if ((self.p1turn and squares[adjacentI] in [2,4,6,8,10]) or (not self.p1turn and squares[adjacentI] in [1,3,5,7,9])):
            moveSpies.append([i,adjacentI])
    return (parachutes, bombs, makeSoldiers, moveSoldiers, makeTanks, moveTanks, makeTrenches, makeSpies, moveSpies)
    
  def endMove(self, newSquares):
    player = self.p1 if self.p1turn==True else self.p2
    if player.turnsToBomb != 0:
      player.turnsToBomb -= 1
    self.p1turn = not self.p1turn
    self.squares = newSquares
 
  def putParachute(self, i):
    squares = self.squares
    p1turn = self.p1turn
    if squares[i] != 0:
      raise Exception('Illegal move parachute')
      return
    if p1turn:
      squares[i]=1
    else:
      squares[i]=2
    self.endMove(squares)
 
  def putBomb(self, i):
    squares = self.squares
    adjacency = self.adjacency
    if squares[i] != 0:
        raise Exception('Illegal move bomb')
        return
    player = self.p1 if self.p1turn==True else self.p2
    if player.turnsToBomb != 0:
      #print(player.turnsToBomb)
      raise Exception('Illegal move bomb')
    player.turnsToBomb = 5
    for squareI in adjacency[i]:
      if squares[squareI] not in [7,8]:
        squares[squareI] = 0
      else:
        squares[squareI]-=6
    self.endMove(squares)
  
  def makeSoldier(self, i):
    squares = self.squares
    if (self.p1turn and squares[i]!=1) or (not self.p1turn and squares[i]!=2):
      raise Exception('Illegal make soldier')
    squares[i]=3 if self.p1turn else 4
    self.endMove(squares)
  
  def moveSoldier(self, originI, destinationI):
    squares = self.squares
    if destinationI not in self.adjacency[originI]:
      raise Exception('moveSoldier: destinationI not adjacent to originI')
    elif (self.p1turn and squares[originI]!=3) or (not self.p1turn and squares[originI]!=4):
      raise Exception('moveSoldier: origin index does not have a soldier')
    elif (squares[destinationI] in [5,6,7,8]):
      raise Exception('moveSoldier: destinationI cannot be killed by soldier')
    squares[originI] = 0
    squares[destinationI] = 3 if self.p1turn else 4
    self.endMove(squares)

  def makeTank(self, i):
    squares = self.squares
    if ((self.p1turn and squares[i]!=3) or (not self.p1turn and squares[i]!=4)):
      raise Exception('makeTank: Illegal move')
    squares[i]=5 if self.p1turn else 6
    self.endMove(squares)

  def getTwoFarAdjacencySet(self,i):
    twoFarAdjacencyFromOriginI = set()
    for adjacentI in self.adjacency[i]:
      twoFarAdjacencyFromOriginI.add(adjacentI)
      for twoFarAdjacentI in self.adjacency[adjacentI]:
        if twoFarAdjacentI!=i:
          twoFarAdjacencyFromOriginI.add(twoFarAdjacentI)
    return twoFarAdjacencyFromOriginI
  
  def moveTank(self,originI,destinationI):
    squares = self.squares
    twoFarAdjacencySet = self.getTwoFarAdjacencySet(originI)
    if destinationI not in twoFarAdjacencySet:
      raise Exception('moveTank: cant move to destination index')
    elif ((self.p1turn and squares[originI]!=5) or (not self.p1turn and squares[originI]!=6)):
      raise Exception('moveTank: no tank on origin square')
    squares[originI] = 0
    squares[destinationI] = 5 if self.p1turn else 6
    if destinationI not in self.adjacency[originI]:
      middleI = (set(self.adjacency[originI]) & set(self.adjacency[destinationI])).pop()
      squares[middleI] = 0
    self.endMove(squares)

  def makeTrench(self,i):
    squares = self.squares
    if (self.p1turn and squares[i]!=1) or (not self.p1turn and squares[i]!=2):
      raise Exception('Illegal make trench')
    squares[i]=7 if self.p1turn else 8
    self.endMove(squares)

  def makeSpy(self,i):
    squares = self.squares
    if ((self.p1turn and squares[i]!=7) or (not self.p1turn and squares[i]!=8)):
      raise Exception('makeSpy: Illegal move. p1turn:{}, square value:{}'.format(self.p1turn,squares[i]))
    squares[i]=9 if self.p1turn else 10
    self.endMove(squares)

  def moveSpy(self,originI,destinationI):
    squares = self.squares
    if destinationI not in self.adjacency[originI]:
      raise Exception('moveSpy: destinationI not adjacent to originI')
    elif (self.p1turn and squares[originI]!=9) or (not self.p1turn and squares[originI]!=10):
      raise Exception('moveSpy: origin index does not have a spy')
    elif (squares[destinationI]==0):
      raise Exception('moveSpy: destination square is empty')
    elif ((self.p1turn and squares[destinationI]%2!=0) or (not self.p1turn and squares[destinationI]%2!=1)):
      raise Exception('moveSpy: spy cannot convert his own teammate')
    squares[destinationI] += -1 if self.p1turn else 1
    self.endMove(squares)
 
  def whoWon(self):
    squares = self.squares
    adjacency = [[0,5,1], \
                [1,7,2], \
                [3,10,13], \
                [5,10,11,15], \
                [7,11,12,17], \
                [9,12,19], \
                [13,20,23], \
                [15,20,21,25], \
                [17,21,22,27], \
                [19,22,29], \
                [25,30,31], \
                [27,31,32]]
    p1bases = 0
    p2bases = 0
    for base in range(len(adjacency)):
        p1squares = 0
        p2squares = 0
        for square in adjacency[base]:
            if squares[square]%2==1:
                p1squares+=1
            elif squares[square]!=0 and squares[square]%2==0:
                p2squares+=1
        if p1squares > len(adjacency[base])/2:
            p1bases+=1
        elif p2squares > len(adjacency[base])/2:
            p2bases+=1
    self.p1bases = p1bases
    self.p2bases = p2bases
    if p1bases>=8:
      return 1
    elif p2bases>=8:
      return 2
    else:
      return None
 
  def printCell(self, baseIndex, i, squares):
    if squares[baseIndex + i] == 0:
        print(' |', end='')
    else:
        print('{}|'.format(squares[baseIndex+i]), end='')
 
  def printBoard(self):
    squares = self.squares
    print('  |', end='')
    baseIndex = 0
    for i in range(3):
        if i==1 or i==2:
            print('X|', end='')
        self.printCell(baseIndex, i, squares)
    print('\n|',end='')
    baseIndex = 3
    for i in range(7):
        self.printCell(baseIndex, i, squares)
    print('\n|',end='')
    print('X|', end='')
    baseIndex = 10
    for i in range(3):
        self.printCell(baseIndex, i, squares)
        print('X|', end='')
    print('\n|',end='')
    baseIndex=13
    for i in range(7):
        self.printCell(baseIndex,i, squares)
    print('\n|',end='')
    print('X|', end='')
    baseIndex=20
    for i in range(3):
        self.printCell(baseIndex, i, squares)
        print('X|', end='')
    print('\n|',end='')
    baseIndex=23
    for i in range(7):
        self.printCell(baseIndex, i, squares)
    print('\n  |',end='')
    baseIndex=30
    for i in range(3):
        if i==1 or i==2:
            print('X|', end='')
        self.printCell(baseIndex,i, squares)
    print()
 
class Player(object):
  def __init__(self):
    self.turnsToBomb = 0
 
MODEL_NAME = 'bruhmodel'
REPLAY_MEMORY_SIZE = 50_000
MINIBATCH_SIZE = 80
DISCOUNT=0.99
BOARD_RANGE=10 #board has range 0-BOARD_RANGE
MIN_REPLAY_MEMORY_SIZE = 1000
UPDATE_TARGET_EVERY = 5
 
class DQNAgent:
  def __init__(self):
    # Main model
    self.model = self.create_model()
    # Target network
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())
    # An array with last n steps for training
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    # Used to count when to update target network with main network's weights
    self.target_update_counter = 0
 
  def create_model(self):
    #make model
    model = Sequential()
    model.add(Dense(21, activation='relu', input_dim=NUMBER_OF_CELLS))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(546, activation='softmax'))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
  
  # Adds step's data to a memory replay array
  # (observation space, action, reward, new observation space, done)
  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)
  
  # Trains main network every step during episode
  def train(self, terminal_state):
    # Start training only if certain number of samples is already saved
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    # Get a minibatch of random samples from memory replay table
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    # Get current states from minibatch, then query NN model for Q values
    current_states = np.array([transition[0] for transition in minibatch])/BOARD_RANGE
    current_qs_list = self.model.predict(current_states)
    # Get future states from minibatch, then query NN model for Q values
    # When using target network, query it, otherwise main network should be queried
    new_current_states = np.array([transition[3] for transition in minibatch])/BOARD_RANGE
    future_qs_list = self.target_model.predict(new_current_states)
    X = []
    y = []
    # Now we need to enumerate our batches
    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
      # If not a terminal state, get new q from future states, otherwise set it to 0
      # almost like with Q Learning, but we use just part of equation here
      if not done:
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + DISCOUNT * max_future_q
      else:
        new_q = reward
      # Update Q value for given state
      current_qs = current_qs_list[index]
      current_qs[action] = new_q
      # And append to our training data
      X.append(current_state)
      y.append(current_qs)
    # Fit on all samples as one batch, log only on terminal state
    self.model.fit(np.array(X)/BOARD_RANGE, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
    # Update target network counter every episode
    if terminal_state:
      self.target_update_counter += 1
    # If counter reaches set value, update target network with weights of main network
    if self.target_update_counter > UPDATE_TARGET_EVERY:
      self.target_model.set_weights(self.model.get_weights())
      self.target_update_counter = 0
 
  # Queries main network for Q values given current observation space (environment state)
  def get_qs(self, state):
    #print(self.model.predict(np.array([state])/BOARD_RANGE)[0])
    return self.model.predict(np.array([state])/BOARD_RANGE)[0]

 
EPISODES = 20_000
AGGREGATE_STATS_EVERY = 50  # episodes
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.001
 
agent1 = DQNAgent()
agent2 = DQNAgent()
ep_rewards1 = []
ep_rewards2 = []
draws = 0
p1wins = 0
p2wins = 0
typeOfUnit = [0]*9
numTurnsInGameY = []
drawY = []
p1winsY = []
p2winsY = []
legalMoves = []
soldierSpyMoveList=[[0, 4], [1, 6], [2, 8], [3, 4], [4, 0], [4, 3], [4, 5], [4, 10], [5, 4], [5, 6], [6, 1], [6, 5], [6, 7], [6, 11], [7, 6], [7, 8], [8, 2], [8, 7], [8, 9], [8, 12], [9, 8], [10, 4], [10, 14], [11, 6], [11, 16], [12, 8], [12, 18], [13, 14], [14, 10], [14, 13], [14, 15], [14, 20], [15, 14], [15, 16], [16, 11], [16, 15], [16, 17], [16, 21], [17, 16], [17, 18], [18, 12], [18, 17], [18, 19], [18, 22], [19, 18], [20, 14], [20, 24], [21, 16], [21, 26], [22, 18], [22, 28], [23, 24], [24, 20], [24, 23], [24, 25], [24, 30], [25, 24], [25, 26], [26, 21], [26, 25], [26, 27], [26, 31], [27, 26], [27, 28], [28, 22], [28, 27], [28, 29], [28, 32], [29, 28], [30, 24], [31, 26], [32, 28]]
tankMoveList=[[0, 10], [0, 3], [0, 4], [0, 5], [1, 11], [1, 5], [1, 6], [1, 7], [2, 8], [2, 9], [2, 12], [2, 7], [3, 0], [3, 10], [3, 4], [3, 5], [4, 0], [4, 3], [4, 5], [4, 6], [4, 10], [4, 14], [5, 0], [5, 1], [5, 3], [5, 4], [5, 6], [5, 7], [5, 10], [5, 11], [6, 1], [6, 4], [6, 5], [6, 7], [6, 8], [6, 11], [6, 16], [7, 1], [7, 2], [7, 5], [7, 6], [7, 8], [7, 9], [7, 11], [7, 12], [8, 2], [8, 6], [8, 7], [8, 9], [8, 12], [8, 18], [9, 8], [9, 2], [9, 12], [9, 7], [10, 0], [10, 3], [10, 4], [10, 5], [10, 13], [10, 14], [10, 15], [10, 20], [11, 1], [11, 5], [11, 6], [11, 7], [11, 15], [11, 16], [11, 17], [11, 21], [12, 2], [12, 7], [12, 8], [12, 9], [12, 17], [12, 18], [12, 19], [12, 22], [13, 10], [13, 20], [13, 14], [13, 15], [14, 4], [14, 10], [14, 13], [14, 15], [14, 16], [14, 20], [14, 24], [15, 10], [15, 11], [15, 13], [15, 14], [15, 16], [15, 17], [15, 20], [15, 21], [16, 6], [16, 11], [16, 14], [16, 15], [16, 17], [16, 18], [16, 21], [16, 26], [17, 11], [17, 12], [17, 15], [17, 16], [17, 18], [17, 19], [17, 21], [17, 22], [18, 8], [18, 12], [18, 16], [18, 17], [18, 19], [18, 22], [18, 28], [19, 17], [19, 18], [19, 12], [19, 22], [20, 10], [20, 13], [20, 14], [20, 15], [20, 23], [20, 24], [20, 25], [20, 30], [21, 11], [21, 15], [21, 16], [21, 17], [21, 25], [21, 26], [21, 27], [21, 31], [22, 32], [22, 12], [22, 17], [22, 18], [22, 19], [22, 27], [22, 28], [22, 29], [23, 24], [23, 25], [23, 20], [23, 30], [24, 14], [24, 20], [24, 23], [24, 25], [24, 26], [24, 30], [25, 20], [25, 21], [25, 23], [25, 24], [25, 26], [25, 27], [25, 30], [25, 31], [26, 16], [26, 21], [26, 24], [26, 25], [26, 27], [26, 28], [26, 31], [27, 32], [27, 21], [27, 22], [27, 25], [27, 26], [27, 28], [27, 29], [27, 31], [28, 32], [28, 18], [28, 22], [28, 26], [28, 27], [28, 29], [29, 32], [29, 27], [29, 28], [29, 22], [30, 24], [30, 25], [30, 20], [30, 23], [31, 25], [31, 26], [31, 27], [31, 21], [32, 27], [32, 28], [32, 29], [32, 22]]

#format:para(<33),bomb(<66-33),soldier(<99-66),moveSoldier(<171,-99),tank(<204,-171),moveTank(<408-204),trench(<441,-408),spy(<474,-441),moveSpy(<546-474)
def tryMovesUntilOneWorksThenMove(game,action):
  global legalMoves,typeOfUnit,soldierSpyMoveList,tankMoveList
  parachuteMoves,bombMoves,makeSoldierMoves,moveSoldierMoves,makeTankMoves,moveTankMoves,makeTrenchMoves,makeSpyMoves,moveSpyMoves = legalMoves
  if action < 33:
    if action in parachuteMoves:
      game.putParachute(action)
      typeOfUnit[0]+=1
      return True
  elif action < 66:
    if (action-33) in bombMoves:
      game.putBomb(action-33)
      typeOfUnit[1]+=1
      return True
  elif action<99:
    if (action-66) in makeSoldierMoves:
      game.makeSoldier(action-66)
      typeOfUnit[2]+=1
      return True
  elif action<171:
    soldierMove = soldierSpyMoveList[action-99]
    if soldierMove in moveSoldierMoves:
      game.moveSoldier(soldierMove[0],soldierMove[1])
      typeOfUnit[3]+=1
      return True
  elif action<204:
    tankMove = action-171
    if tankMove in makeTankMoves:
      game.makeTank(tankMove)
      typeOfUnit[4]+=1
      return True
  elif action<408:
    moveTankMove = tankMoveList[action-204]
    if moveTankMove in moveTankMoves:
      game.moveTank(moveTankMove[0],moveTankMove[1])
      typeOfUnit[5]+=1
      return True
  elif action<441:
    trenchMove = action-408
    if trenchMove in makeTrenchMoves:
      game.makeTrench(trenchMove)
      typeOfUnit[6]+=1
      return True
  elif action<474:
    spyMove = action-441
    if spyMove in makeSpyMoves:
      game.makeSpy(spyMove)
      typeOfUnit[7]+=1
      return True
  elif action<546:
    moveSpyMove = soldierSpyMoveList[action-474]
    if moveSpyMove in moveSpyMoves:
      game.moveSpy(moveSpyMove[0],moveSpyMove[1])
      typeOfUnit[8]+=1
      return True
  return False

def move(game, agent, agentnum, epsilon):
  #Before Move
  global legalMoves
  done = False
  current_state = game.squares
  current_bases = game.p1bases if agentnum==1 else game.p2bases
  other_bases = game.p2bases if agentnum==1 else game.p1bases
  legalMoves = game.getLegalMoves(1 if game.p1turn else 2)

  #Making a move
  if np.random.random() > epsilon:
    # Get action from Q table
    #print('model')
    actions = np.argsort(agent.get_qs(current_state))[::-1]
    for action in actions:
      if tryMovesUntilOneWorksThenMove(game,action):
        break
  else:
    #Random
    #print('random')
    randLow=0
    randHigh=545
    while 1==1:
      action = np.random.randint(randLow, randHigh)
      #print(action, end=" ")
      if tryMovesUntilOneWorksThenMove(game,action):
        break

  #After Move
  #game.printBoard()
  new_state = game.squares
  new_bases = game.p1bases if agentnum==1 else game.p2bases
  new_other_bases = game.p2bases if agentnum==1 else game.p1bases
  #check if draw
  legalMoves = game.getLegalMoves(1 if game.p1turn else 2)
  numPossibleMoves = 0
  for typeOfMove in legalMoves:
    numPossibleMoves += len(typeOfMove)
  winner = game.whoWon()
  reward = 0
  if winner is None:
    if numPossibleMoves==0:
      done=True
    elif new_bases<current_bases:
      reward = -0.1
    elif new_bases>current_bases:
      reward = 0.1
  elif winner==agentnum:
    done=True
    reward = 1
  elif winner and winner != agentnum:
    done=True
  # Every step we update replay memory and train main network
  agent.update_replay_memory((current_state, action, reward, new_state, done))
  agent.train(done)
  return done
 
# Iterate over episodes
for episode in range(EPISODES):
    p1 = Player()
    p2 = Player()
    game = Game(p1,p2)
    # Restarting episode - reset episode reward and step number
    step = 0
    # Reset environment and get initial state
    current_state = game.squares
    
    # Reset flag and start iterating until episode ends
    done = False
    while not done:
      done = move(game,agent1,1,epsilon)
      if done:
        break
      done = move(game,agent2,2,epsilon)
      step +=2
    winner = game.whoWon()
    if winner is None:
      episode_reward = (0,0)
    elif winner==1:
      episode_reward=(1,0)
    else:
      episode_reward=(0,1)
    
    # Decay epsilon for p1
    if epsilon > MIN_EPSILON:
      epsilon *= EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)
    
    #save model2
    ep_rewards2.append(episode_reward[1])
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        win_percent = sum(ep_rewards2[-AGGREGATE_STATS_EVERY:])/len(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        # Save model, but only when min reward is greater or equal a set value
        agent2.model.save('./models/{}__p2__win_percent{}__gamenum{}.model'.format(MODEL_NAME, win_percent,draws+p1wins+p2wins))
    
    #print
    if episode_reward == (0,0):
      draws+=1
    elif episode_reward == (1,0):
      p1wins+=1
    elif episode_reward == (0,1):
      p2wins+=1
    drawY.append(draws)
    p1winsY.append(p1wins)
    p2winsY.append(p2wins)
    numTurnsInGameY.append(step)

plt.plot(np.arange(EPISODES),drawY)
plt.savefig('draws.png')
plt.plot(np.arange(EPISODES),p1winsY)
plt.savefig('p1wins.png')
plt.plot(np.arange(EPISODES),p2winsY)
plt.savefig('p2wins.png')
plt.plot(np.arange(EPISODES),numTurnsInGameY)
plt.savefig('gameTurns.png')
