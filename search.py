# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import heapq
import queue
from abc import ABC, abstractmethod

import util
from util import Queue


class SearchProblem(ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abstractmethod
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    @abstractmethod
    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    from util import Stack
    start = problem.getStartState()
    frontier = Stack()
    frontier.push((start, []))  # cada entrada: (estado, lista_de_acciones)
    visitados = set()

    while not frontier.isEmpty():
        estado, camino = frontier.pop()

        if estado in visitados:
            continue
        visitados.add(estado)

        if problem.isGoalState(estado):
            return camino

        for sucesor, accion, _ in problem.getSuccessors(estado):
            if sucesor not in visitados:
                frontier.push((sucesor, camino + [accion]))
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue

    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontera = Queue()
    frontera.push((start, []))  # (estado, camino_de_acciones)
    visitados = set([start])


    while not frontera.isEmpty():
        estado, camino = frontera.pop()

        if problem.isGoalState(estado):
            return camino

        for hijo, accion, costeAccion in problem.getSuccessors(estado):
            if hijo not in visitados:
                visitados.add(hijo)
                frontera.push((hijo, camino + [accion]))


    return []





import heapq

def uniformCostSearch(problem):
    """
    Returns a list of actions that reaches the goal with minimum path cost.
    Assumes:
      - problem.getStartState()
      - problem.isGoalState(state) -> bool
      - problem.getSuccessors(state) -> (successor, action, stepCost)
      - problem.getCostOfActions(actions) -> float
      ESTARIA ESTE MINI ESQUEMA EN EL ENUNCIADO YA QUE ME HA COSTADO ENTENDER LA ESTRUCTURA DE LA CLASE PROBLEMA :>
    """
    estado_inicial = problem.getStartState()
    # cola ordenada automaticamente por prioridad
    cola_prioridad = [(0.0, estado_inicial, [])]
    # mejor coste por estado
    mejor_coste = {estado_inicial: 0.0}

    while cola_prioridad:
        coste_Actual, estado_actual, acciones = heapq.heappop(cola_prioridad)

        # Saltamos las entradas que no sean utiles
        if coste_Actual > mejor_coste.get(estado_actual, float("inf")):
            continue

        # verificamos si estamos en la solucion
        if problem.isGoalState(estado_actual):
            return acciones

        # Estudiamos cada hijo del nodo X y lo añadimos si procede
        for estados_hijos, accion, coste_p in problem.getSuccessors(estado_actual):

            candidate_cost = coste_Actual + coste_p
            if candidate_cost < mejor_coste.get(estados_hijos, float("inf")):
                mejor_coste[estados_hijos] = candidate_cost

                heapq.heappush(cola_prioridad,(candidate_cost, estados_hijos, acciones + [accion]))

    return []





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    # ESTO SIRVE PARA HACER EL A* SIN HEURISTICO EN EL CASO DE QUERER USAR O MHTAN O EUC ESTAN YA IMPLEMENTADOS *
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue

    start = problem.getStartState()
    frontier = PriorityQueue()                # elementos: (estado, camino, g)
    frontier.push((start, [], 0), 0 + heuristic(start, problem))

    mejor_costo = {start: 0}                  # g*(estado) = mejor coste conocido hasta estado
    visitados = set()                         # opcional: cerrados para evitar reprocesar

    while not frontier.isEmpty():
        estado, camino, g = frontier.pop()

        # Entrada obsoleta: si existe un g mejor registrado, se ignora
        if g > mejor_costo.get(estado, float('inf')):
            continue

        if problem.isGoalState(estado):
            return camino

        if estado in visitados:
            continue
        visitados.add(estado)

        for sucesor, accion, paso in problem.getSuccessors(estado):
            nuevo_g = g + paso
            if nuevo_g < mejor_costo.get(sucesor, float('inf')):
                mejor_costo[sucesor] = nuevo_g
                f = nuevo_g + heuristic(sucesor, problem)
                frontier.push((sucesor, camino + [accion], nuevo_g), f)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
