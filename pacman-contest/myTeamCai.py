# myTeamCai.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from game import Actions
import game
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import os


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='ApproximateQLearningAgent', second='ApproximateQLearningAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex, "offensive"), eval(second)(secondIndex, "defensive")]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
    Your initialization code goes here, if you need any.
    '''

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)

        '''
    You should change this in your own agent.
    '''

        return random.choice(actions)


class ReinforcementAgent(CaptureAgent):
    """
  This is a basic reinforcement learning Agent
  """
    def __init__(self, index, epsilon=0.10, gamma=0.8, alpha=0.20):
        super().__init__(index)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)


    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

    def chooseAction(self, gameState):
        util.raiseNotDefined()


class ApproximateQLearningAgent(ReinforcementAgent):
    def __init__(self, index, flag):
        super().__init__(index)
        self.actions = [Directions.EAST, Directions.WEST, Directions.SOUTH, Directions.NORTH, Directions.STOP]
        self.flag = flag
        # 13 features
        self.feature_vector = np.zeros(13, dtype=float)
        # 5 actions * 13 features (might change)
        # order: East, West, South, North, Stop
        self.weight_matrix = np.zeros((5, 13), dtype=float)
        self.qValue = np.zeros(5)
        self.last_qValue = None
        self.last_feature_vector = None
        self.initial_foodnum = None
        self.win = False
        self.enmeyWin = False

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

        if self.flag == "offensive":
            # initial feature vector for offensive
            if os.path.isfile('./weightvector_offensive.save'):
                # file read bytes
                file = open("weightvector_offensive.save", "rb")
                self.weight_matrix = pickle.load(file)
            else:
                # file write bytes
                file = open("weightvector_offensive.save", "wb")
                pickle.dump(self.weight_matrix, file)
        else:
            # initial feature vector for defensive
            if os.path.isfile('./weightvector_defensive.save'):
                # file read bytes
                file = open("weightvector_defensive.save", "rb")
                self.weight_matrix = pickle.load(file)
            else:
                # file write bytes
                file = open("weightvector_defensive.save", "wb")
                pickle.dump(self.weight_matrix, file)

        self.initial_foodnum = len(self.getFood(gameState).asList())


    def chooseAction(self, gameState):
        # if self.flag == "defensive":
        #     print(self.weight_matrix)
        legal_actions = gameState.getLegalActions(self.index)
        # print(legal_actions)
        # extract current features
        self.feature_vector = self.extractFeatures(gameState)
        # print(self.feature_vector)

        # compute current Q value using current feature vector
        self.computeQvalue()
        # if self.flag == "defensive":
        #     print(self.qValue)
        # choose action for this round
        action = self.computeActionFromQvalue(legal_actions)
        # print(action)

        self.last_qValue = self.qValue.copy()
        self.last_feature_vector = self.feature_vector.copy()
        # get new state and reward then update weight vectors
        newGameState = gameState.generateSuccessor(self.index, action)
        reward = self.getReward(gameState, newGameState, action)
        # if self.flag == "offensive":
        #     print(reward)
        # if self.flag == "defensive":
        #     print(reward)



        # do the update
        self.update(newGameState, reward, action)

        return action

    def computeQvalue(self):
        for i in range(len(self.qValue)):
            self.qValue[i] = np.dot(self.weight_matrix[i], self.feature_vector)

    def computeQvalueForOneAction(self, index):
        self.qValue[index] = np.dot(self.weight_matrix[index], self.feature_vector)

    # using epsilon greedy and tie breaker
    def computeActionFromQvalue(self, legal_actions):
        if np.random.rand(1) > self.epsilon:
            # get available indices of the action
            legal_indices = self.get_indices_of_legal_actions(legal_actions)
            # get available q value
            legal_Q_value = np.array([self.qValue[index] for index in legal_indices])

            max_indices = np.where(legal_Q_value == np.max(legal_Q_value))[0]
            index = np.random.choice(max_indices)
            action_index = legal_indices[index]
            return self.actions[action_index]
        else:
            return np.random.choice(legal_actions)

    def update(self, newGameState, reward, action):
        action_index = self.actions.index(action)
        legal_actions = newGameState.getLegalActions(self.index)

        loop_index = 0
        for action in self.actions:
            if action in legal_actions:
                self.feature_vector = self.extractFeatures(newGameState.generateSuccessor(self.index, action))
                self.computeQvalueForOneAction(loop_index)
            else:
                # treat action as STOP
                self.feature_vector = self.extractFeatures(newGameState.generateSuccessor(self.index, Directions.STOP))
                self.computeQvalueForOneAction(loop_index)
            loop_index += 1

        difference = self.alpha * (reward + self.gamma * np.max(self.qValue) - self.last_qValue[action_index])
        self.weight_matrix[action_index] = np.add(self.weight_matrix[action_index], difference*self.last_feature_vector)

        # save updated weight to files
        if self.flag == "offensive":
            file = open("weightvector_offensive.save", "wb")
            pickle.dump(self.weight_matrix, file)
        else:
            file = open("weightvector_defensive.save", "wb")
            pickle.dump(self.weight_matrix, file)


    # get rewards after applying action
    def getReward(self, gameState, newGameState, action):
        my_old_position = gameState.getAgentPosition(self.index)
        my_new_position = newGameState.getAgentPosition(self.index)

        my_old_state = gameState.getAgentState(self.index)
        my_new_state = newGameState.getAgentState(self.index)

        my_total_food = my_new_state.numCarrying + my_new_state.numReturned
        enemy_total_food = 0

        enemy_indices = self.getOpponents(gameState)
        enemy_pacman = []
        enemy_pacman_new = []

        enmey_pacman_eating_food = False
        enemy_pacman_return_food = 0
        enemy_win = False

        eating_scared_ghost = False
        enemy_pacman_got_eaten = False

        for index in enemy_indices:
            old_agentState = gameState.getAgentState(index)
            new_agentState = newGameState.getAgentState(index)
            enemy_total_food += new_agentState.numCarrying
            enemy_total_food += new_agentState.numReturned
            # for enemy that is pacman
            if old_agentState.isPacman:
                if new_agentState.numCarrying > old_agentState.numCarrying:
                    enmey_pacman_eating_food = True
                if new_agentState.numReturned + new_agentState.numCarrying >= self.initial_foodnum - 2:
                    enemy_win = True
                enemy_pacman_return_food = new_agentState.numReturned - old_agentState.numReturned

                pacman_pos = gameState.getAgentPosition(index)
                new_pacman_pos = newGameState.getAgentPosition(index)
                if pacman_pos is not None:
                    enemy_pacman.append(pacman_pos)
                    if self.getMazeDistance(my_old_position, pacman_pos) < 2 and not new_agentState.isPacman and enemy_pacman_return_food == 0:
                        enemy_pacman_got_eaten = True
                if new_pacman_pos is not None:
                    enemy_pacman_new.append(new_pacman_pos)


            # for enemy that is ghost
            else:
                # for scared ghost
                if old_agentState.scaredTimer > 0:
                    old_ghost_pos = gameState.getAgentPosition(index)
                    new_ghost_pos = newGameState.getAgentPosition(index)
                    # test if eating scared ghost
                    if old_ghost_pos is not None:
                        if new_ghost_pos is not None:
                            if self.getMazeDistance(my_old_position, old_ghost_pos) < 2 and self.getMazeDistance(old_ghost_pos, new_ghost_pos) > 1:
                                eating_scared_ghost = True
                        else:
                            eating_scared_ghost = True

        reward = 0.0
        # for offensive agent
        if self.flag == "offensive":
            # useful data
            old_food_list = self.getFood(gameState).asList()
            old_capsule_list = self.getCapsules(gameState)
            num_new_returned = my_new_state.numReturned - my_old_state.numReturned

            closest_distance_to_food_old = min([self.getMazeDistance(my_old_position, food) for food in old_food_list])
            closest_distance_to_food_new = min([self.getMazeDistance(my_new_position, food) for food in old_food_list])
            # time punishment and approach to food
            if closest_distance_to_food_new < closest_distance_to_food_old:
                reward += 1.0
            else:
                reward -= 1.0

            # if eating food +5
            if len(old_food_list) > 0:
                if my_new_position in old_food_list:
                    reward += 5.0
            # if returning n food + n*5
            if not num_new_returned == 0:
                reward += 5.0 * num_new_returned

            # if win +20
            if my_new_state.numReturned + my_new_state.numCarrying >= self.initial_foodnum - 2 and self.win == False:
                reward += 20.0
                self.win = True
            elif gameState.data.timeleft <= 1 and my_total_food > enemy_total_food:
                reward += 20.0
            elif gameState.data.timeleft <= 1 and my_total_food < enemy_total_food:
                reward -= 20.0

            # eating capsule + 5.5
            if len(old_capsule_list) > 0:
                if my_new_position in old_capsule_list:
                    reward += 5.5

            # if eating scared ghost + 5.5
            if eating_scared_ghost:
                reward += 5.5

            # if got killed - 5.5
            if self.getMazeDistance(my_old_position, my_new_position) > 1:
                reward -= 10.0

        # for defensive agent
        else:
            # useful data
            old_capsule_list = self.getCapsulesYouAreDefending(gameState)
            old_food_list = self.getFoodYouAreDefending(gameState).asList()

            # if not seeing pacman, then try to reach the farest food
            if len(enemy_pacman) == 0:
                if not len(old_food_list) == 0:
                    farest_before = max([self.getMazeDistance(my_old_position, food) for food in old_food_list])
                    farest_after = max([self.getMazeDistance(my_new_position, food) for food in old_food_list])
                    if farest_before > farest_after:
                        reward += 1.0
            # if seeing pacman, try to reach pacman
            else:
                min_distance_before = min(self.getMazeDistance(my_old_position, pacman) for pacman in enemy_pacman)
                min_distance_after = min(self.getMazeDistance(my_new_position, pacman) for pacman in enemy_pacman)

                if min_distance_before > min_distance_after:
                    reward += 1.0
                elif min_distance_before < min_distance_after:
                    reward -= 1.0

            # if enemy pacman eta food -5
            if enmey_pacman_eating_food:
                reward -= 5.0

            # if enemy pacman return food -7.5
            if enemy_pacman_return_food:
                reward -= 7.5

            # if enemy win -20, if i win +20
            if enemy_win and self.enmeyWin == False:
                reward -= 20.0
                self.enmeyWin = True
            elif gameState.data.timeleft <= 1 and my_total_food > enemy_total_food:
                reward += 20.0
            elif gameState.data.timeleft <= 1 and my_total_food < enemy_total_food:
                reward -= 20.0

            # if pacman eat scared me -10
            if my_old_state.scaredTimer > 0:
                if self.getMazeDistance(my_old_position, my_new_position) > 1:
                    reward -= 10.0

            # if pacman eat capsule -5.5
            if len(old_capsule_list) > 0 and len(enemy_pacman_new) > 0:
                for enemy in enemy_pacman_new:
                    if enemy in old_capsule_list:
                        reward -= 5.5

            # if i eat pacman +20
            if enemy_pacman_got_eaten:
                reward += 20.0

        return reward


    def get_indices_of_legal_actions(self, legal_actions):
        indices = []
        for action in legal_actions:
            indices.append(self.actions.index(action))
        return indices

    def getOpponentsDistances(self, gameState):
        return [gameState.getAgentDistances()[opponentIndex]
                for opponentIndex in self.getOpponents(gameState)]


    # This function extract features from the current game state
    # return a np array of 13 features
    def extractFeatures(self, gameState):
        """
        For offensive agent:
        features are:
            bias term
            distance to the closest food
            number of active ghost one step away
            number of active ghost two step away
            distance to the closest active ghost
            inverse of the distance to the closest active ghost
            number of scared ghosts one step away
            number of scared ghost two step away
            distance to closest scared ghost
            minimum distance to a capsule
            activation of capsule if active ghost nearby
            activation of capsule if scared ghost nearby
            eating foods if there are no active ghost nearby
        For deffensive agent:
        features are:
            bias term
            distance to the closest pacman, if not exist, then distance to the closest ghost else 0
            shortest distance of pacman to the nearest food
            distance of pacman to the nearest capsule
            number of food of the pacman is carrying
            number of food returned
            number of pacman in two steps
            number of pacman in one step
            number of food remaining
            number of pacman within 1 step of me if im scared
            umber of pacman within 2 step of me if im scared
            eating pacman if i'm not scared
            remaining scared time

        :param gameState: current game layout
        :return: feature vector
        """
        # initial arrays to store feature
        features = []

        if self.flag == "offensive":

            # get my position
            myPosition = gameState.getAgentPosition(self.index)

            # get foods postion
            food_matrix = self.getFood(gameState)
            foods = convert_food_matrix(food_matrix)

            # get capsule position
            capsules = self.getCapsules(gameState)

            # get enemy position
            enemy_indice = self.getOpponents(gameState)
            enemy_active = []
            enemy_scared = []
            enemy_active_distance = []
            enemy_scared_distance = []
            capsules_activited = False
            for index in enemy_indice:
                agentState = gameState.getAgentState(index)
                if not agentState.isPacman:
                    e_position = gameState.getAgentPosition(index)
                    if not e_position is None:
                        if agentState.scaredTimer == 0:
                            enemy_active.append(e_position)
                        else:
                            enemy_scared.append(e_position)
                            capsules_activited = True
                    # if enemy not in sight
                    else:
                        # if active
                        if agentState.scaredTimer == 0:
                            enemy_active_distance.append(gameState.getAgentDistances()[index])
                        else:
                            enemy_scared_distance.append(gameState.getAgentDistances()[index])

            # bias term
            features.append(1)

            # distance to the closest food
            food_distance = min([self.getMazeDistance(myPosition, food) for food in foods])
            features.append(food_distance)

            # number of active ghost one and two step away
            # distance to the closest active ghost
            oneStep_active = 0
            twoStep_active = 0
            minDistance_active_enemy = 10
            if len(enemy_active) > 0:
                enemy_distance = [self.getMazeDistance(myPosition, enemy_position) for enemy_position in enemy_active]
                minDistance_active_enemy = min(enemy_distance)
                for d in enemy_distance:
                    if d == 1:
                        oneStep_active += 1
                    if d == 2:
                        twoStep_active += 1
            elif len(enemy_active_distance) > 0:
                minDistance_active_enemy = min(enemy_active_distance)
            features.append(oneStep_active)
            features.append(twoStep_active)
            features.append(minDistance_active_enemy)

            # inverse of the distance to the closest active ghost
            if minDistance_active_enemy == 0:
                features.append(50)
            else:
                features.append(1 / minDistance_active_enemy)
            # number of scared ghosts one step away
            # number of scared ghost two step away
            # distance to closest scared ghost
            oneStep_scared = 0
            twoStep_scared = 0
            minDistance_scared_enemy = 10
            if len(enemy_scared) > 0:
                enemy_scared_distance = [self.getMazeDistance(myPosition, enemy_position) for enemy_position in
                                         enemy_scared]
                minDistance_scared_enemy = min(enemy_scared_distance)
                for d in enemy_scared_distance:
                    if d == 1:
                        oneStep_scared += 1
                    if d == 2:
                        twoStep_scared += 1
            elif len(enemy_scared_distance) > 0:
                minDistance_scared_enemy = min(enemy_scared_distance)
            features.append(oneStep_scared)
            features.append(twoStep_scared)
            features.append(minDistance_scared_enemy)

            # minimum distance to a capsule
            if len(capsules) > 0:
                capsule_minDistance = min([self.getMazeDistance(myPosition, capsule_position) for capsule_position in capsules])
                features.append(capsule_minDistance)
            else:
                features.append(0)

            # activation of capsule if active ghost nearby
            capsule_activation_active = 0
            if capsules_activited and (oneStep_active > 0 or twoStep_active > 0):
                capsule_activation_active = 1
            features.append(capsule_activation_active)

            # activation of capsule if scared ghost nearby
            capsule_activation_scared = 0
            if capsules_activited and (oneStep_scared > 0 or twoStep_scared > 0):
                capsule_activation_scared = 1
            features.append(capsule_activation_scared)

            # eating foods if there are no active ghost nearby
            pre_gameState = self.getPreviousObservation()
            eating_food_safe = 0

            if pre_gameState is not None:
                pre_food_matrix = self.getFood(pre_gameState)
                if pre_food_matrix[myPosition[0]][myPosition[1]] and oneStep_active == 0 and twoStep_active == 0:
                    eating_food_safe = 1
            features.append(eating_food_safe)

        else:
            # get my position
            myPosition = gameState.getAgentPosition(self.index)
            myAgentState = gameState.getAgentState(self.index)

            my_scared_time = myAgentState.scaredTimer

            # get food defending postion
            food_matrix = self.getFoodYouAreDefending(gameState)
            food_list = convert_food_matrix(food_matrix)

            # get pacman and ghost position
            # get food carrying, food returned
            enemy_indice = self.getOpponents(gameState)
            pacman = []
            ghost = []
            food_carrying = 0
            food_returned = 0
            num_pacman_onestep = 0
            num_pacman_two_step = 0
            ghost_distance = []
            pacman_distance = []
            for index in enemy_indice:
                agentState = gameState.getAgentState(index)
                if agentState.isPacman:
                    food_carrying += agentState.numCarrying
                    food_returned += agentState.numReturned

                    p_position = gameState.getAgentPosition(index)
                    if p_position is not None:
                        pacman.append(p_position)
                        distance = self.getMazeDistance(myPosition, p_position)
                        if distance == 1:
                            num_pacman_onestep += 1
                        if distance == 2:
                            num_pacman_two_step += 1
                    # if pacman not in sight
                    else:
                        pacman_distance.append(gameState.getAgentDistances()[index])
                else:
                    food_returned += agentState.numReturned

                    p_position = gameState.getAgentPosition(index)
                    if p_position is not None:
                        ghost.append(p_position)
                    else:
                        ghost_distance.append(gameState.getAgentDistances()[index])

            # get capsule defending
            capsules = self.getCapsulesYouAreDefending(gameState)

            # bias term 1
            features.append(1)

            # distance to the closest pacman, if not exist, then distance to the closest ghost
            if len(pacman) > 0:
                min_distance_pacman = min([self.getMazeDistance(myPosition, pacman_pos) for pacman_pos in pacman])
                features.append(min_distance_pacman)
            elif len(pacman_distance) > 0:
                features.append(min(pacman_distance))
            elif len(ghost_distance) > 0:
                features.append(min(ghost_distance))
            else:
                features.append(0)

            # shortest distance of pacman to the nearest food
            if len(pacman) > 0:
                min_distance_list = []
                if len(food_list) > 0:
                    for p in pacman:
                        min_distance = min([self.getMazeDistance(p, food) for food in food_list])
                        min_distance_list.append(min_distance)
                    features.append(min(min_distance_list))
                else:
                    features.append(0)
            elif len(ghost) > 0:
                min_distance_list = []
                if len(food_list) > 0:
                    for g in ghost:
                        min_distance = min([self.getMazeDistance(g, food) for food in food_list])
                        min_distance_list.append(min_distance)
                    features.append(min(min_distance_list))
                else:
                    features.append(0)
            else:
                if len(pacman_distance) > 0:
                    features.append(min(pacman_distance))
                elif len(ghost_distance) > 0:
                    features.append(min(ghost_distance))
                else:
                    features.append(0)

            # distance of pacman to the nearest capsule
            if len(pacman) > 0 :
                if len(capsules) > 0:
                    min_distance_list = []
                    for p in pacman:
                        min_distance = min([self.getMazeDistance(p, capsule) for capsule in capsules])
                        min_distance_list.append(min_distance)
                    features.append(min(min_distance_list))
                else:
                    features.append(0)
            elif len(ghost) > 0:
                if len(capsules) > 0:
                    min_distance_list = []
                    for g in ghost:
                        min_distance = min([self.getMazeDistance(g, capsule) for capsule in capsules])
                        min_distance_list.append(min_distance)
                    features.append(min(min_distance_list))
                else:
                    features.append(0)
            else:
                if len(capsules) > 0:
                    if len(pacman_distance) > 0:
                        features.append(min(pacman_distance))
                    elif len(ghost_distance) > 0:
                        features.append(min(ghost_distance))
                    else:
                        features.append(0)
                else:
                    features.append(0)

            # number of food of the pacman is carrying
            features.append(food_carrying)

            # number of food returned
            features.append(food_returned)

            # number of pacman in two steps
            features.append(num_pacman_two_step)

            # number of pacman in one step
            features.append(num_pacman_onestep)

            # number of food remaining
            features.append(len(food_list))

            # number of pacman within 1 step of me if im scared
            # number of pacman within 2 step of me if im scared
            scared_one_step = 0
            scared_two_step = 0
            if my_scared_time > 0 and len(pacman) > 0:
                for p in pacman:
                    distance = self.getMazeDistance(myPosition, p)
                    if distance == 1:
                        scared_one_step += 1
                    if distance == 2:
                        scared_two_step += 1

            features.append(scared_one_step)
            features.append(scared_two_step)

            # eating pacman if i'm not scared
            eating = 0
            if my_scared_time == 0 and len(pacman) > 0:
                for p in pacman:
                    if p == myPosition:
                        eating = 1
            features.append(eating)

            # remaining scared time
            features.append(my_scared_time)


        n = Normalizer()
        features = np.array(features).reshape((1, -1))
        new = n.fit_transform(features)
        return new[0]

#####################################################################################################################
# Helper Function
#####################################################################################################################



def convert_food_matrix(food_matrix):
    return food_matrix.asList()