# myTeam.py
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


import random
import time
from random import choice

import __main__
import distanceCalculator
import util
from captureAgents import CaptureAgent
from game import Directions, Actions
from util import nearestPoint
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import os


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='AStarAgent', second='AStarAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class AStarAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def getClosestHomeLine(self, gameState):
        if self.red:
            homeLine = [(gameState.data.layout.width // 2 - 2, y) for y in
                        range(0, gameState.data.layout.height)]
        else:
            homeLine = [(gameState.data.layout.width // 2 + 1, y) for y in
                        range(0, gameState.data.layout.height)]
        return [a for a in homeLine if a not in gameState.getWalls().asList()]

    def printCurrentAgentState(self, gameState, printGameState=False):
        """
        Debugging code
        :param gameState:
        :param printGameState:
        :return:
        """
        print("Agent " + str(self.index) +
              " [Red = " + str(self.red) + "] [isOffensive=" + str(
            self.isOffensive) + "] [Pacman=" +
              str(self.isAgentAPacman(gameState, self.index)) + "]：")

        if printGameState:
            print("Game state: ")
            print(gameState)

        teamIndex = self.getTeam(gameState)
        print("Team Agents Index: ")
        print(teamIndex)
        for agentIndex in teamIndex:
            print("Team Agent[" + str(agentIndex) + "]'s position = " + str(
                self.getAgentPosition(gameState, agentIndex)))

        opponentsIndex = self.getOpponents(gameState)
        print("Opponent Team Agents Index: ")
        print(opponentsIndex)
        # print("Opponent scared timer: ")
        # print(self.opponentsScaredTimer)
        for agentIndex in opponentsIndex:
            try:
                print("Opponent Agent[" + str(
                    agentIndex) + "]'s state = " + str(
                    self.getAgentState(gameState, agentIndex)))
            except:
                print("Fail to get Opponent Agent[" + str(
                    agentIndex) + "]'s state")
            # try:
            #     print("Opponent Agent[" + str(
            #         agentIndex) + "]'s scaredTimer = " + str(
            #         self.getAgentState(gameState, agentIndex)).scaredTimer)
            # except:
            #     print("Fail to get Opponent Agent[" + str(
            #         agentIndex) + "]'s scared timer")

            try:
                print("Opponent Agent[" + str(
                    agentIndex) + "]'s position = " + str(
                    self.getAgentPosition(gameState, agentIndex)))
            except:
                print("Fail to get Opponent Agent[" + str(
                    agentIndex) + "]'s position")

        print("Opponent positions: ")
        print(self.getOpponentsPosition(gameState))

        print("Current agent scared timer = " + str(
            self.getAgentScaredTime(gameState, self.index)))
        print("Current agent is scared = " + str(
            self.getAgentScaredTime(gameState, self.index) > 0))
        print("self.isOffensiveDueToScared = " + str(
            self.isOffensiveDueToScared))
        print("self.stopCount = " + str(self.stopCount))
        print("CurrentScore: " + str(self.getScore(gameState)))
        print("Total food: " + str(self.totalNumberOfFood))
        print("Food left: " + str(len(self.getFood(gameState).asList())))
        print("Total defending food: " + str(self.totalNumberOfDefendingFood))
        print("Defending food left: " + str(
            len(self.getFoodYouAreDefending(gameState).asList())))
        print("Capsule: " + str(len(self.getCapsules(gameState))))
        print("Defending capsule left: " + str(
            len(self.getCapsulesYouAreDefending(gameState))))
        print("Food carrying: " + str(
            self.getNumberOfFoodCurrentAgentCarrying(gameState)))
        print("Opponents positions():")
        print(self.getOpponentsDistances(gameState))
        # print(self.distanceDictionary)
        print("map size: " + str(gameState.data.layout.width) + "x" + str(
            gameState.data.layout.height))
        # print(gameState.data.layout.height)
        # print("Previous observation:")
        # print(self.getPreviousObservation())
        # print("Current observation:")
        # print(self.getCurrentObservation())
        print("\n")

    def calculateDistanceBetweenPoint(self, gameState, distanceOnGrid=False):
        """
        Pre-calculate the distance between points, so we do not need to
        calculate it during the execution, to speed up the decision-making
        process.
        """
        # Map of the game
        layout = list(set(self.getFoodList(gameState) +
                          self.getFood(gameState).asList(False)))

        # Wall of current map.
        walls = gameState.getWalls().asList()
        walkablePositions = list(set(
            [(x, y) for (x, y) in layout if (x, y) not in walls]
        ))

        # Store distance between points as a dictionary of dictionary,
        # which looks like: {position_i: {position_j: distance_between_i_j}}.

        if distanceOnGrid:
            return {position: {
                targetPosition:
                    self.distancer.getDistanceOnGrid(position, targetPosition)
                for targetPosition in walkablePositions}
                for position in walkablePositions}
        else:
            return {position: {
                targetPosition: self.getMazeDistance(position, targetPosition)
                for targetPosition in walkablePositions}
                for position in walkablePositions}

    def getFoodList(self, gameState):
        """
        Return the remaining food in the provided gameState as a list of
        coordinates.
        :param gameState: game state to be queried.
        :return: A list of coordinates
        """
        return self.getFood(gameState).asList()

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.data.layout)
        self.distancer.getMazeDistances()
        if '_display' in dir(__main__):
            self.display = __main__._display
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.walls = gameState.getWalls().asList()
        self.home = gameState.getAgentState(self.index).getPosition()
        self.food = self.getFoodList(gameState)
        self.defendingFood = self.getFoodYouAreDefending(gameState).asList()

        # flag indicating if agent is offensive
        self.isOffensive = self.index > 1
        # flag indicating if agent is offensive due to scared
        self.isOffensiveDueToScared = False
        # How many 'Stop' action has agent being taken in a row.
        self.stopCount = 0
        self.totalNumberOfFood = len(self.food)
        self.totalNumberOfDefendingFood = len(
            self.getFoodYouAreDefending(gameState).asList())

        start = time.time()
        self.distanceDictionary = self.calculateDistanceBetweenPoint(gameState)
        print('eval time for agent %d: %.4f' %
              (self.index, time.time() - start))

        # The following attribute is for offensive agent
        self.home = gameState.getAgentState(self.index).getPosition()

        # The following attributes are for defensive agent
        self.initial_state = self.initialStateFoodList(gameState)
        self.foodEaten = []
        self.ateFood = None

        # history_action_list
        self.position_list = []
        self.last_action = None

        self.unAvaliable_foodList = []
        self.currentgoal = None

        # randomly choose a middle point and drive agent there.
        if self.index > 1:
            middleLines = self.getUpperHalfRandomMiddleLinePos()
        else:
            middleLines = self.getLowerHalfRandomMiddleLinePos()
        self.destination = random.choice(middleLines)


    def getTeammateIndex(self, gameState):
        currentTeam = self.getTeam(gameState)
        return [agentIndex for agentIndex in currentTeam
                if agentIndex != self.index][0]

    def getSurroundingCoordinatesOfTheCoordinate(self, targetingCoordinate, distance=5):
        """

        :param targetingCoordinate: targeting coordinate, in the form (x, y).
        :param distance:
        :return:
        """
        if targetingCoordinate is None:
            return None

        targetingIntCoordinate = self.convertPositionToIntPosition(targetingCoordinate)
        distancesToOtherCoordinate = self.distanceDictionary[targetingIntCoordinate]
        return [coordinate for coordinate, distanceToCoordinate in distancesToOtherCoordinate.items()
                if distanceToCoordinate <= distance]

    def getMiddleLines(self):
        if self.red:
            middle_line = [((self.width / 2) - 1, y) for y in
                           range(0, self.height)]
        else:
            middle_line = [(self.width / 2, y) for y in range(0, self.height)]
        available_middle = [a for a in middle_line if a not in self.walls]
        return available_middle

    # def getRandomMiddleLine(self):
    #     middleLines = self.getMiddleLines()
    #     return choice(middleLines)

    def getClosestMiddle(self, gameState):
        agentPos = self.getAgentPosition(gameState, self.index)
        middleLines = self.getMiddleLines()
        middleLinesDistances = {
            middlePos: self.getMazeDistance(agentPos, middlePos) for
            middlePos in middleLines}

        closestMiddleLineDistance = min(middleLinesDistances.values())

        closestDistanceMiddleLine = [locationCoordinate
                                     for (locationCoordinate, distance) in
                                     middleLinesDistances.items() if
                                     distance == closestMiddleLineDistance]

        if len(closestDistanceMiddleLine) > 0:
            return random.choice(closestDistanceMiddleLine)
        else:
            return None

    def getClosestMiddlePositionForOpponent(self, opponentPos):
        middleLines = self.getMiddleLines()
        middleLinesDistances = {
            middlePos: self.getMazeDistance(opponentPos, middlePos) for
            middlePos in middleLines}

        closestMiddleLineDistance = min(middleLinesDistances.values())

        closestDistanceMiddleLine = [locationCoordinate
                                     for (locationCoordinate, distance) in
                                     middleLinesDistances.items() if
                                     distance == closestMiddleLineDistance]

        if len(closestDistanceMiddleLine) > 0:
            return random.choice(closestDistanceMiddleLine)
        else:
            return None

    def getOpponentPacmansPosition(self, gameState):
        opponentPacmanPositions = []
        opponentIndex = self.getOpponentsIndex(gameState)
        for e in opponentIndex:
            if gameState.getAgentState(e).isPacman:
                e_pos = gameState.getAgentPosition(e)
                if e_pos is not None:
                    opponentPacmanPositions.append(e_pos)
        if len(opponentPacmanPositions) > 0:
            return opponentPacmanPositions
        else:
            return None

    def manhattanDistance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def getOpponentsIndex(self, gameState):
        return self.getOpponents(gameState)

    def getOpponentGhostsPosition(self, gameState, isGhost=True):
        return self.getOpponentGhostsPositionFromState(
            self.getOpponentGhostsState(gameState, isGhost))
        # opponentsGhost = self.getOpponentGhostsState(gameState, isGhost)
        # if opponentsGhost is not None:
        #     return [opponentGhostState.getPosition() for opponentGhostState in
        #             self.getOpponentGhostsState(gameState, isGhost)]
        # else:
        #     return None

    def getOpponentGhostsPositionFromState(self, opponentGhost):
        if opponentGhost is not None:
            return [opponentGhostState.getPosition() for opponentGhostState in
                    opponentGhost]
        else:
            return None

    def getOpponentGhostsState(self, gameState, isGhost=True):
        opponentsIndex = self.getOpponentsIndex(gameState)
        opponentsState = [gameState.getAgentState(opponentIndex) for
                          opponentIndex in opponentsIndex]
        opponentsGhostState = [opponentState
                               for opponentState in opponentsState
                               if opponentState.getPosition() is not None and
                               (not opponentState.isPacman == isGhost)]

        if len(opponentsGhostState) != 0:
            return opponentsGhostState
        else:
            return None

    def getUnscaredOpponentGhostsPosition(self, gameState):
        opponentsIndex = self.getOpponentsIndex(gameState)
        opponentsState = [gameState.getAgentState(opponentIndex) for
                          opponentIndex in opponentsIndex]
        opponentsGhostState = [opponentState
                               for opponentState in opponentsState
                               if opponentState.getPosition() is not None and
                               not opponentState.isPacman and
                               opponentState.scaredTimer == 0]

        if len(opponentsGhostState) != 0:
            return [
                self.convertPositionToIntPosition(opponentState.getPosition())
                for opponentState in opponentsGhostState]
        else:
            return None

    def getClosestFood(self, gameState):
        """
        Get the closest food's coordinate
        :param gameState: game state to be queried.
        :return: closest food' coordinate.
        """
        print("self.getFoodList(gameState):")
        print(self.getFoodList(gameState))
        return self.getClosestFoodFromList(gameState,
                                           self.getFoodList(gameState))

    def getClosestDefendingFood(self, gameState):
        """
        Get the closest food's coordinate
        :param gameState: game state to be queried.
        :return: closest food' coordinate.
        """
        return self.getClosestFoodFromList(gameState,
                                           self.getFoodYouAreDefending(
                                               gameState).asList())

    def getClosestFoodFromList(self, gameState, foodList):
        """
        Get the closest food's coordinate
        :param gameState: game state to be queried.
        :return: closest food' coordinate.
        """
        if foodList is None or len(foodList) == 0:
            return None
        available_food = foodList.copy()
        print("sssssssssssssself.checkRepeatAction()")
        print(self.checkRepeatAction())

        if self.checkRepeatAction():
            if self.currentgoal != None:
                print("sssssssssssssssssssself.currentgoal")
                print(self.currentgoal)
                self.position_list = []
                available_food = [f for f in foodList if f != self.currentgoal]

        print("available_food")
        print(available_food)

        agentPos = self.getAgentPosition(gameState, self.index)

        foodDistances = {foodPos: self.getMazeDistance(agentPos, foodPos)
                         for foodPos in available_food}
        closestFoodDistance = min(foodDistances.values())

        closestDistanceFoods = [
            foodCoordinate for (foodCoordinate, distance) in
            foodDistances.items() if distance == closestFoodDistance]

        if len(closestDistanceFoods) > 0:
            return random.choice(closestDistanceFoods)
        else:
            return None

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


    def getAvailableFood(self, gameState):
        if self.unAvaliable_foodList == [] and self.getFoodList(gameState) == None:
            return self.getClosestMiddle(gameState)
        if self.getFoodList(gameState) != None and self.unAvaliable_foodList != []:
            available = list(set(self.getFoodList(gameState)).difference(set(self.availableFoodList)))
            availableFood = self.getClosestFoods(available)
        else:
            self.unAvaliable_foodList == []
        return availableFood

    # def getClosestFoods(self, gameState, available):
    #     return self.getClosestFoodFromList(gameState, available)

    def availableFoodList(self, gameState, goal):
        if self.getFoodList(gameState) == None:
            return None
        if self.checkRepeatAction():
            if goal in self.getFoodList(gameState):
                self.unAvaliable_foodList.append(goal)
        return self.unAvaliable_foodList

    # def getMazeDistance(self, pos1, pos2):
    #     d = self.distancer.getDistance(pos1, pos2)
    #     return d

    def normalHeuristic(self, pos1, pos2):
        return 0

    def getNumberOfFoodAgentCarrying(self, gameState, agentIndex):
        """
        Get the number of food agent is currently carrying
        :param gameState: game state to be queried.
        :return: the number of food agent is carrying
        """
        return self.getAgentState(gameState, agentIndex).numCarrying

    def getNumberOfFoodCurrentAgentCarrying(self, gameState):
        """
        Get the number of food agent is currently carrying
        :param gameState: game state to be queried.
        :return: the number of food agent is carrying
        """
        return self.getNumberOfFoodAgentCarrying(gameState, self.index)

    def getAgentPosition(self, gameState, agentIndex):
        """
        Get specified agent's position.
        :param gameState: the current game state
        :param agentIndex: agent whose position to be queried
        :return: the position of current agent.
        """
        agentPosition = self.getAgentState(gameState, agentIndex).getPosition()
        if agentPosition is not None:
            return self.convertPositionToIntPosition(agentPosition)
        return None

    def convertPositionToIntPosition(self, agentPosition):
        if agentPosition is None:
            return None

        xFloat, yFloat = agentPosition
        return int(xFloat), int(yFloat)

    def isAgentCloseToMiddleLine(self, gameState, agentIndex, expectedDistanceToMiddle=2):
        """
        Check if the agent is close enough to the middle line.
        :param gameState: current game state
        :param agentIndex: agent to be checked
        :param expectedDistanceToMiddle: expected distance, by default it's 2
        :return:
        """
        agentPos = self.getAgentPosition(gameState, agentIndex)
        middleLines = self.getMiddleLines()
        if agentPos is None or middleLines is None or len(middleLines) == 0:
            return False

        distancesToMiddle = [self.distanceDictionary[agentPos][middleLinePos] for middleLinePos in middleLines]
        minDistanceToMiddle = min(distancesToMiddle)

        return minDistanceToMiddle <= expectedDistanceToMiddle

    def splitMiddleLineIntoTwo(self):
        """
        Split the middle line position list into 2 list, ordered by the y-axis value.
        :return:
        """
        middleLines = self.getMiddleLines()
        if middleLines is None or len(middleLines) == 0:
            return None

        # Sort middle lines by y.
        sortedMiddleLines = sorted(middleLines, key=lambda tup: tup[1])
        middleValIndex = len(sortedMiddleLines) // 2

        return sortedMiddleLines[0: middleValIndex], sortedMiddleLines[middleValIndex:]

    def getUpperHalfRandomMiddleLinePos(self):
        upperHalf, lowerHalf = self.splitMiddleLineIntoTwo()
        return upperHalf

    def getLowerHalfRandomMiddleLinePos(self):
        upperHalf, lowerHalf = self.splitMiddleLineIntoTwo()
        return lowerHalf

    def aStarSearch(self, gameState, goal, heuristic):
        goalAsIntCoordinate = self.convertPositionToIntPosition(goal)
        # print("A-star entered")
        # print("goal: ")
        # print(goal)
        # print("goalAsIntCoordinate: ")
        # print(goalAsIntCoordinate)
        # print(gameState)

        start = self.getAgentPosition(self.getCurrentObservation(), self.index)

        open = util.PriorityQueue()
        open.push((start, []), 0)
        node = []
        while not open.isEmpty():
            coordinate, path = open.pop()
            if coordinate == goalAsIntCoordinate:
                if len(path) == 0:
                    return 'Stop'
                # print("path")
                # print(path)
                print("\n")
                return path[0]
            if coordinate not in node:
                node.append(coordinate)
                for nodes in self.getSuccessors(coordinate, gameState):
                    cost = len(path + [nodes[1]]) + heuristic(start, nodes[0])
                    if nodes not in node:
                        open.update((nodes[0], path + [nodes[1]]), cost)

        # if goal != self.home:
        #     return self.aStarSearch(gameState, self.home, self.manhattanDistance)
        # print("line 394 reached.")
        return 'Stop'

    def getSuccessors(self, coordinate, gameState):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST,
                       Directions.WEST]:
            x, y = coordinate
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls:
                nextPosition = (nextx, nexty)
                if self.getUnscaredOpponentGhostsPosition(
                        gameState) is not None:

                    opponentGhostsPosition = self.getOpponentGhostsPosition(gameState)
                    if opponentGhostsPosition is not None and len(opponentGhostsPosition) > 0:
                        if nextPosition not in self.getOpponentGhostsPosition(gameState):
                        # if nextPosition not in self.getSurroundingCoordinatesOfTheCoordinate(opponentGhostsPosition[0]):
                            successors.append((nextPosition, action))
                else:
                    successors.append((nextPosition, action))
        return successors

    def getCloseCapsule(self, gameState):
        capsule = self.getCapsules(gameState)
        if len(capsule) == 0:
            return None
        else:
            distance = []
            for c in capsule:
                dis = self.manhattanDistance(
                    self.getAgentPosition(gameState, self.index), c)
                distance.append(dis)
            return min(distance)

    def getClosestCapsulePos(self, gameState):
        capsules = self.getCapsules(gameState)
        if len(capsules) != 0:

            currentAgentPos = self.getAgentPosition(gameState, self.index)
            capsulesDistanceDict = {
                capsulePos: self.getMazeDistance(currentAgentPos, capsulePos)
                for capsulePos in capsules}
            minCapsuleDistance = min(capsulesDistanceDict.values())
            closestCapsule = [capsulePos for capsulePos, capsuleDistance in
                              capsulesDistanceDict.items() if
                              capsuleDistance == minCapsuleDistance]
            if len(closestCapsule) > 0:
                return random.choice(closestCapsule)
        return None

    def getAgentScaredTime(self, gameState, agentIndex):
        return self.getAgentState(gameState, agentIndex).scaredTimer

    def getOpponentGhostsIndex(self, gameState):
        """
        Get opponents' ghosts index.
        :param gameState: the game state to be queried.
        :return: A list of int indicating the opponents ghosts' index.
        """
        opponentGhostsIndex = []
        for d in self.getOpponentsIndex(gameState):
            if not self.isAgentAPacman(gameState, d):
                opponentGhostsIndex.append(d)
        return opponentGhostsIndex

    def isAgentAPacman(self, gameState, agentIndex):
        """
        Get agent's status.
        :param gameState: the current game state
        :param agentIndex: agent whose position to be queried
        :return: True if it is a pacman, and False if it's ghost.
        """
        return self.getAgentState(gameState, agentIndex).isPacman

    def getOpponentsPosition(self, gameState):
        """
        Return the opponents position as dictionary, with the opponent index
        as the key, and position/None as the value.
        :param gameState: the current game state
        :return: opponents positions dictionary
        """
        opponents = self.getOpponents(gameState)
        return {opponent: self.getAgentPosition(gameState, opponent)
                for opponent in opponents
                }

    def getOpponentsDistances(self, gameState):
        """
        Return the distances to opponents, if opponent is observable, return
        the exact distance, otherwise return the noisy distance.
        :param gameState: the current game state
        :return: A list of int, which indicates distance of current agent to
        opponent.
        """
        agentPosition = self.getAgentPosition(gameState, self.index)

        opponentsPosition = self.getOpponentsPosition(gameState)
        opponentsDistanceList = [
            self.getMazeDistance(opponentPosition, agentPosition)
            for opponentPosition in opponentsPosition.values()
            if opponentPosition is not None]

        if len(opponentsDistanceList) > 0:
            return opponentsDistanceList
        else:
            return [gameState.getAgentDistances()[opponentIndex]
                    for opponentIndex in self.getOpponents(gameState)]

    def getAgentState(self, gameState, agentIndex):
        """
        Get agent's state in the provided game state.
        :param gameState: game state to be queried.
        :param agentIndex: index of agent to be queried.
        :return: state of the specified agent in the provided game state.
        """
        return gameState.getAgentState(agentIndex)

    def wasAgentKilled(self, agentIndex):
        """
        Check if the specified agent is killed.
        :param agentIndex: agent whose status is going to be checked.
        :return: True if agent was killed in the last round, otherwise False.
        """
        previousGameState = self.getPreviousObservation()
        currentGameState = self.getCurrentObservation()

        if previousGameState is not None:
            oldAgentPosition = self.getAgentPosition(previousGameState,
                                                     agentIndex)
            # print(oldAgentPosition)

            if currentGameState is not None:
                currentAgentPosition = self.getAgentPosition(
                    currentGameState, agentIndex)
                # print(currentAgentPosition)
                # print("distance moved")
                # print(self.distanceDictionary[currentAgentPosition][oldAgentPosition])

                return self.getMazeDistance(currentAgentPosition, oldAgentPosition) > 1
        return False

    def updateStopCount(self, action):
        if action == "Stop":
            self.stopCount += 1
        else:
            self.stopCount = 0

    # def unavailableFood(self, action, gameState):
    #     if action == "Stop":
    #

    def chooseAction(self, gameState):
        start = time.time()

        if self.isAgentCloseToMiddleLine(gameState, self.index) or not self.isOurAgentHome(gameState):
            self.destination = None

        if self.destination is not None:
            return self.aStarSearch(gameState, self.destination, self.manhattanDistance)

        myPos = self.getAgentPosition(gameState, self.index)

        self.isOffensiveDueToScared = gameState.getAgentState(self.index).scaredTimer > 0
        print(self.isOffensiveDueToScared)

        # if i'm killed and teamate is not and not distance within 2, then switch role
        wasKilled = self.wasAgentKilled(self.index)
        isTeammateKilled = self.wasAgentKilled(self.getTeammateIndex(gameState))
        team_distance = self.getTeamateDistance(gameState)
        if (isTeammateKilled and not wasKilled and team_distance > 3) or \
                (wasKilled and not isTeammateKilled and team_distance > 3):
            self.isOffensive = not self.isOffensive

        # if repeat action occur, random choice legal action
        # possible improvements: example all possible action, return action that have high value

        # if self.checkRepeatAction():
        #     legal_action = gameState.getLegalActions(self.index)
        #     legal_positions = self.getLegalPosition(myPos, legal_action)
        #     final_positions = [pos for pos in legal_positions if pos != myPos]
        #     final_actions = self.getNewActions(myPos, final_positions)
        #     self.position_list = []
        #     action = random.choice(final_actions)
        #     self.last_action = action
        #     return action

        self.printCurrentAgentState(gameState)

        if self.isOffensive:
            # if self.isOffensive:
            action = self.chooseOffensiveAction(gameState)
        else:
            action = self.chooseDefensiveAction(gameState)

        self.updateStopCount(action)  # update the stop count
        # print('eval time for agent %d to choose action: %.4f\n\n' %
        #       (self.index, time.time() - start))

        x0, y0 = Actions.directionToVector(action)
        x, y = myPos
        self.position_list.append((x + x0, y + y0))
        self.last_action = action

        return action

    def getTeamateDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        teammatePos = gameState.getAgentPosition(self.getTeammateIndex(gameState))

        return self.getMazeDistance(myPos, teammatePos)

    def checkRepeatAction(self):
        if len(self.position_list) < 5:
            return False
        else:
            counter = util.Counter()
            for position in self.position_list:
                counter[position] += 1

            for key in counter.keys():
                if counter[key] >= 3:
                    return True
                else:
                    self.position_list = []
        return False

    def getLegalPosition(self, myPos, actions):
        positions = []
        x, y = myPos
        for action in actions:
            x0, y0 = Actions.directionToVector(action)
            new_x = x + x0
            new_y = y + y0
            positions.append((new_x, new_y))
        return positions

    def getNewActions(self, myPos, final_positions):
        actions = []
        x, y = myPos
        for pos in final_positions:
            x0, y0 = pos
            direction = Actions.vectorToDirection((x0 - x, y0 - y))
            if direction != self.last_action:
                actions.append(direction)
        return actions

    def isOurAgentHome(self, gameState):
        agentPosX, agentPosY = gameState.getAgentPosition(self.index)
        middlePointX, middlePointY = self.getClosestMiddle(gameState)
        if self.red:
            return agentPosX <= middlePointX
        else:
            return agentPosX >= middlePointX

    def isOpponentInOurHome(self, gameState, agentIndex):
        agentPosX, agentPosY = gameState.getAgentPosition(agentIndex)
        middlePointX, middlePointY = self.getClosestMiddle(gameState)
        if self.red:
            return agentPosX > middlePointX
        else:
            return agentPosX < middlePointX

    def chooseOffensiveAction(self, gameState):
        pacman_pos = self.getOpponentPacmansPosition(gameState)
        # print("self.isAgentCloseToMiddleLine(gameState, self.inedx)")
        # print(self.isAgentCloseToMiddleLine(gameState, self.index))
        # print("self.splitMiddleLineIntoTwo()")
        # print(self.splitMiddleLineIntoTwo())
        # print("self.getUpperHalfRandomMiddleLinePos()")
        # print(self.getUpperHalfRandomMiddleLinePos())
        # print("self.getLowerHalfRandomMiddleLinePos()")
        # print(self.getLowerHalfRandomMiddleLinePos)

        currentAgentPos = gameState.getAgentPosition(self.index)
        closestMiddlePos = self.getClosestMiddle(gameState)
        closestFoodPos = self.getClosestFood(gameState)

        # go back home if time is limited.
        if gameState.data.timeleft < 150 and \
                self.getNumberOfFoodCurrentAgentCarrying(gameState) >= self.totalNumberOfDefendingFood / 5:
            print("2")
            print("time to back home")
            self.currentgoal = closestMiddlePos
            return self.aStarSearch(gameState,
                                    closestMiddlePos,
                                    self.manhattanDistance)

        print("3")
        opponentGhostsState = self.getOpponentGhostsState(gameState)
        opponentGhostsPosition = self.getOpponentGhostsPositionFromState(
            opponentGhostsState)

        print("opponentGhostsPosition")
        print(opponentGhostsPosition)

        opponentsIndex = self.getOpponentsIndex(gameState)
        for opponentIndex in opponentsIndex:
            print("opponentIndex = " + str(opponentIndex))

            if opponentIndex is not None:
                foodOpponentCarrying = self.getNumberOfFoodAgentCarrying(
                    gameState, opponentIndex)
                print("foodOpponentCarrying=" + str(foodOpponentCarrying))

                opponentPos = self.getAgentPosition(gameState, opponentIndex)

                print("opponentPos=" + str(opponentPos))

                if opponentPos is not None:
                    opponentClosestMiddlePoint = self.getClosestMiddlePositionForOpponent(
                        opponentPos)
                    opponentDistanceToMiddle = \
                        self.getMazeDistance(opponentClosestMiddlePoint, opponentPos)
                    offensiveAgentDistanceToOpponent = \
                        self.getMazeDistance(opponentPos, currentAgentPos)
                    # if chased by opponent, choose go back or eat the capsule.
                    if foodOpponentCarrying >= \
                            self.totalNumberOfDefendingFood // 4 and \
                            offensiveAgentDistanceToOpponent <= \
                            2 * opponentDistanceToMiddle:
                        action = self.aStarSearch(gameState, opponentPos,
                                                  self.manhattanDistance)
                        print("808 action")
                        print(action)
                        return action

        if opponentGhostsPosition is not None and \
                not self.isOurAgentHome(gameState):
            if len(self.getCapsules(gameState)) != 0:
                # if there is a capsule.
                oppnentGhostPoints = self.getOpponentGhostsPosition(gameState)
                enemy_distance = [self.getMazeDistance(self.convertPositionToIntPosition(currentAgentPos), enemy) for
                                  enemy in oppnentGhostPoints]
                distanceToOpponent = min(enemy_distance)
                enemy_index = enemy_distance.index(distanceToOpponent)
                enemy_point = oppnentGhostPoints[enemy_index]

                if distanceToOpponent <= 4:
                    if self.distanceDictionary[currentAgentPos][self.getClosestCapsulePos(gameState)] < self.distanceDictionary[enemy_point][self.getClosestCapsulePos(gameState)]:
                        action = self.aStarSearch(gameState, self.getClosestCapsulePos(gameState), self.getMazeDistance)

                        return action
                    if self.isOurAgentHome(gameState) and self.stopCount >= 1:
                        print("3.1.1")
                        randomMiddleLinePos = self.getRandomMiddleLine()
                        action = self.aStarSearch(gameState,
                                                  randomMiddleLinePos,
                                                  self.manhattanDistance)
                    else:
                        print("3.1.2")
                        print("3.1.2")
                        action = self.aStarSearch(gameState,
                                                  closestMiddlePos,
                                                  self.manhattanDistance)
                    print("838 action")
                    print(action)

                    return action

                print("3.2")
                if self.distanceDictionary[currentAgentPos][enemy_point] < 4:
                    action = self.aStarSearch(gameState,
                                              self.getUpperHalfRandomMiddleLinePos(),
                                              self.getMazeDistance)
                else:
                    action = self.aStarSearch(gameState, closestFoodPos, self.getMazeDistance)
                    self.currentgoal = closestFoodPos
                print("action")
                print(action)
                return action
            else:
                # if capsule do not exists, and chased by defender, choose go
                # back
                print("4")
                for d in self.getOpponentGhostsPosition(gameState):

                    if self.getOpponentGhostsIndex(gameState):
                        for defe in self.getOpponentGhostsIndex(gameState):
                            if self.getAgentScaredTime(gameState, defe) > 5:
                                print("4.1")
                                self.currentgoal = closestFoodPos
                                return self.aStarSearch(gameState,
                                                        closestFoodPos,
                                                        self.manhattanDistance)
                    print("4.2")
                    if self.manhattanDistance(
                            currentAgentPos, d) < 5:
                        return self.aStarSearch(gameState,
                                                closestMiddlePos,
                                                self.manhattanDistance)

        print("5")
        if len(self.getFoodList(gameState)) == 0:
            return self.aStarSearch(gameState,
                                    closestMiddlePos,
                                    self.manhattanDistance)
        print("6")
        if self.getOpponentGhostsPosition(gameState) is not None and \
                not self.isOurAgentHome(gameState):
            for d in self.getOpponentGhostsIndex(gameState):
                # print("d:")
                # print(d)
                if self.getAgentScaredTime(gameState, d) > 5:
                    self.currentgoal = closestFoodPos
                    return self.aStarSearch(gameState,
                                            closestFoodPos,
                                            self.manhattanDistance)
            return self.aStarSearch(gameState,
                                    closestMiddlePos,
                                    self.manhattanDistance)
        # print("7")
        if pacman_pos is not None:
            for opponentPacmanPosition in pacman_pos:
                if opponentPacmanPosition is not None and \
                        self.getMazeDistance(currentAgentPos, opponentPacmanPosition) < 2:
                    return self.aStarSearch(gameState,
                                            opponentPacmanPosition,
                                            self.manhattanDistance)
        # print("8")
        # go back home if the food agent is carrying is more than 1/4 of the
        # total food.

        if gameState.getAgentState(self.index).numCarrying >= int(
                len(self.food) * 0.5):
            closestFood = self.getClosestFood(gameState)
            if closestFood is not None and \
                    self.getMazeDistance(closestFood, currentAgentPos) > 2:
                return self.aStarSearch(gameState, closestMiddlePos,
                                        self.manhattanDistance)
        # print("self.getOpponentGhostsPosition(gameState)")
        # print(self.getOpponentGhostsPosition(gameState))
        # print("self.isOurAgentHome(gameState)")
        # print(self.isOurAgentHome(gameState))
        if self.getOpponentGhostsPosition(gameState) is not None and \
                self.isOurAgentHome(gameState):
            print("11")
            oppnentGhostPoints = self.getOpponentGhostsPosition(gameState)
            enemy_distance = [self.getMazeDistance(self.convertPositionToIntPosition(currentAgentPos), enemy) for
                              enemy in oppnentGhostPoints]
            distanceToOpponent = min(enemy_distance)
            enemy_index = enemy_distance.index(distanceToOpponent)
            enemy_point = oppnentGhostPoints[enemy_index]
            # print("enemy_point")
            # print(enemy_point)
            # print("currentAgentPos")
            # print(currentAgentPos)
            print("self.isAgentAPacman((gameState, self.index))")
            print(self.isAgentAPacman(gameState, self.index))
            if not self.isAgentAPacman(gameState, self.index) and self.distanceDictionary[currentAgentPos][enemy_point] < 5:

                return self.aStarSearch(gameState, self.home, self.manhattanDistance)

        print("1")
        print("closestFoodPos")
        print(closestFoodPos)
        self.currentgoal = closestFoodPos
        return self.aStarSearch(gameState, closestFoodPos,
                                self.normalHeuristic)

    # The following action are for defensive agent
    def getLatestEatenFoodPos(self, gameState):
        currentFood = self.getFoodYouAreDefending(gameState).asList()
        if self.getPreviousObservation() is not None:
            lastFood = self.getFoodYouAreDefending(
                self.getPreviousObservation()).asList()
            if len(currentFood) < len(lastFood):
                self.foodEaten.append(
                    list(set(lastFood).difference(set(currentFood))))
        if self.foodEaten:
            return self.foodEaten[-1]

        return None

    def initialStateFoodList(self, gameState):
        current_food = self.getFoodYouAreDefending(gameState).asList()
        return current_food

    def hasOpponentEatenFood(self):
        if self.getPreviousObservation() is not None:
            lastFood = self.getFoodYouAreDefending(
                self.getPreviousObservation()).asList()
            return self.totalNumberOfDefendingFood > len(lastFood)

        return False

    def getRandomMiddleLine(self):
        middleLines = self.getMiddleLines()
        return random.choice(middleLines)

    def getEatenFoodChange(self):
        last_state = self.getPreviousObservation()
        current_state = self.getCurrentObservation()

        if last_state is not None:
            last_food_remain = len(self.getFoodYouAreDefending(last_state).asList())
            current_food_remain = len(self.getFoodYouAreDefending(current_state).asList())

            if current_food_remain != last_food_remain:
                return True
        return False

    def keepDistance(self, gameState, pacman_pos):
        myPos = self.getAgentPosition(gameState, self.index)
        distances = [self.getMazeDistance(myPos, pac) for pac in pacman_pos]
        minDistance = min(distances)
        min_index = distances.index(minDistance)
        pacpos = pacman_pos[min_index]

        # if too close, go to the position before
        if minDistance < 4:
            myPreviousState = self.getPreviousObservation()
            myPreviousPos = self.getAgentPosition(myPreviousState, self.index)
            return self.aStarSearch(gameState, myPreviousPos, self.manhattanDistance)
        # else, follow pacman
        else:
            return self.aStarSearch(gameState, pacpos, self.manhattanDistance)

    def chooseDefensiveAction(self, gameState):
        pacman_pos = self.getOpponentPacmansPosition(gameState)
        myPos = self.getAgentPosition(gameState, self.index)

        print("defensive 1")
        # opponent eaten food
        if self.hasOpponentEatenFood():
            print("defensive 2")
            # opponent in sight
            if pacman_pos is not None:
                distances = [self.getMazeDistance(myPos, pac) for pac in pacman_pos]
                minDistance = min(distances)
                min_index = distances.index(minDistance)
                pacpos = pacman_pos[min_index]

                # if im scared: keep distance
                if self.isOffensiveDueToScared:
                    return self.keepDistance(gameState, pacman_pos)

                print("defensive 3")
                # go to the ghost position
                return self.aStarSearch(gameState,
                                        pacpos,
                                        self.manhattanDistance)
            print("defensive 4")
            # opponent not in sight
            lastEatenFoodList = self.getLatestEatenFoodPos(gameState)
            if lastEatenFoodList is not None:
                lastEatenFoodPos = lastEatenFoodList[0]
                print("last eaten food pos = " + str(lastEatenFoodList))
                self.lastEatenDefendingFoodPos = lastEatenFoodPos
                print("defensive 5")

                # arrive last eaten food position
                if myPos == self.lastEatenDefendingFoodPos:
                    print("defensive 9")
                    # enemy not eaten food
                    if not self.getEatenFoodChange():
                        return self.aStarSearch(gameState,
                                                self.getClosestMiddle(gameState),
                                                self.manhattanDistance)
                    else:
                        return self.aStarSearch(gameState,
                                                self.getClosestDefendingFood(
                                                    gameState),
                                                self.manhattanDistance)

                print("defensive 10")
                return self.aStarSearch(gameState, lastEatenFoodPos,
                                        self.manhattanDistance)

        print("defensive 6")
        # enemy in sight and no food has been eaten
        if pacman_pos is not None:

            # if im scared: keep distance
            if self.getAgentScaredTime(gameState, self.index) > 1:
                return self.keepDistance(gameState, pacman_pos)
            print("defensive 7")
            return self.aStarSearch(gameState,
                                    pacman_pos[0],
                                    self.manhattanDistance)

        # print("defensive 8")
        # if gameState.getAgentPosition(self.index) == self.ateFood:
        #     print("defensive 9")
        #     return self.aStarSearch(gameState, self.getClosestFood(gameState),
        #                             self.manhattanDistance)

        print("defensive 10")
        if self.getAgentScaredTime(gameState, self.index) > 0:
            print("defensive 11")
            return self.aStarSearch(gameState, self.getClosestCapsulePos(gameState),
                                    self.manhattanDistance)

        # initially go to middle position
        print("defensive 12")
        return self.aStarSearch(gameState, self.getRandomMiddleLine(),
                                self.manhattanDistance)


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
