from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import threading
import time
import ctypes


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if len(env.passengers) > 0:
             has_gas_to_0 = self.hasGasToTarget(env, taxi_id, env.passengers[0].position)
        else:
            has_gas_to_0 = 0
        if len(env.passengers) > 1:
            has_gas_to_1 = self.hasGasToTarget(env, taxi_id, env.passengers[1].position)
        else:
            has_gas_to_1 = 0
        if taxi.passenger != None:
            target_dest = taxi.passenger.destination
            has_gas_to_2 = self.hasGasToTarget(env, taxi_id, target_dest)
            profit = manhattan_distance(taxi.passenger.position, taxi.passenger.destination)
        else:
            target_dest = taxi.position
            has_gas_to_2 = 0
            profit = 0
        should_refuel = self.shouldRefuel(env, taxi_id, max(self.calculateProfit(env, taxi_id, 0), self.calculateProfit(env, taxi_id, 1), 0), has_gas_to_0, has_gas_to_1, has_gas_to_2)         
        return max(has_gas_to_0 * self.calculateProfit(env, taxi_id, 0), has_gas_to_1 * self.calculateProfit(env, taxi_id, 1)) + has_gas_to_2 * (taxi.fuel - manhattan_distance(taxi.position, target_dest)) * profit + (taxi.fuel - manhattan_distance(taxi.position, env.gas_stations[self.getClosestGasStation(taxi, env)].position)) * should_refuel + taxi.cash


    def calculateProfit(self, env: TaxiEnv, taxi_id, passenger_id):
        if passenger_id == 1:
            if len(env.passengers) < 2:
                return 0
        if len(env.passengers) > 0:
            s = env.passengers[passenger_id].position
            d = env.passengers[passenger_id].destination
            distance_to_travel = manhattan_distance(s, d) + manhattan_distance(env.get_taxi(taxi_id).position, s)
            profit = manhattan_distance(s, d) - manhattan_distance(env.get_taxi(taxi_id).position, s)
            return profit * (distance_to_travel + 2 <= env.num_steps / 2 + 1)
        return 0

    def hasGasToTarget(self, env, taxi_id, target):
        taxi = env.get_taxi(taxi_id)
        if manhattan_distance(taxi.position, target) > taxi.fuel:
            return 0
        return 1

    def getClosestGasStation(self, taxi, env: TaxiEnv):
        dist0 = manhattan_distance(taxi.position, env.gas_stations[0].position)
        dist1 = manhattan_distance(taxi.position, env.gas_stations[1].position)
        if dist0 < dist1:
            return 0
        return 1

    def getBestPassenger(self, taxi, env: TaxiEnv, taxi_id):
        p0 = 0
        p1 = 0
        if len(env.passengers) > 0:
            p0 = self.calculateProfit(env, taxi_id, 0)
        if len(env.passengers) > 1:
            p1 = self.calculateProfit(env, taxi_id, 1)
        if p0 > p1:
            return 0
        return 1

    def shouldRefuel(self, env: TaxiEnv, taxi_id, max_profit, gas_0, gas_1, gas_2):
        if (gas_0 + gas_1 + gas_2 > 0):
            return 0
        taxi = env.get_taxi(taxi_id)
        remaining_steps = env.num_steps
        return min(remaining_steps / 2 + 1 - 2, env.get_taxi(taxi_id).cash) >= max_profit and max_profit > env.get_taxi((taxi_id + 1) % 2).cash


    def canReachToPassengerAndRefuel(self, env: TaxiEnv, taxi, passenger_id):
        distance_to_destination = manhattan_distance(taxi.position, env.passengers[passenger_id].position) + manhattan_distance(env.passengers[passenger_id].destination, env.passengers[passenger_id].position)
        distance_to_gas_station = min(manhattan_distance(env.gas_stations[0].position, env.passengers[passenger_id].destination), manhattan_distance(env.gas_stations[1].position, env.passengers[passenger_id].destination))
        return taxi.fuel >= distance_to_destination + distance_to_gas_station


global minimax_step
minimax_step = ""

class AgentMinimax(AgentGreedyImproved):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.end_time = time.time() + 0.8*time_limit
        self.id = agent_id
        # step = self.id_minimax(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.id_minimax(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def id_minimax(self, env, agent_id):
        depth = 1
        step = 0
        last_chosen_step = -1000000
        while True: #depth < 5?
            last_chosen_step = max(last_chosen_step, self.minimax(env, agent_id, depth))
            if time.time() > self.end_time:
                print("max depth: ", depth)
                break 
            else:
                step = last_chosen_step
            depth += 1
        # print('Got to depth of: ', depth)
        return step

    # def minimax(self, env, agent_id, l):
    #     operators = env.get_legal_operators(agent_id)
    #     children = [env.clone() for _ in operators]
    #     for child, op in zip(children, operators):
    #         child.apply_operator(agent_id, op)
    #     children_results = [self.min(child, 1-agent_id, l-1) for child in children]
    #     max_result = max(children_results)
    #     index_selected = children_results.index(max_result)
    #     return operators[index_selected]
    #   
    def minimax(self, env, agent_id, l):
        if l == 0 or (env.get_taxi(agent_id).fuel == 0 and env.get_taxi(1 - agent_id).fuel == 0):
            return self.heuristic(env, agent_id)
        if time.time() > self.end_time:
            return -1
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        if agent_id == self.id:
            children_max_results = [self.minimax(child, 1 - agent_id, l - 1) for child in children]
            cur_max = max(children_max_results)
            return cur_max
        else:
                
            children_min_results = [self.minimax(child, 1-agent_id, l-1) for child in children]
            cur_min = min(children_min_results)
            # index_selected = children_results.index(max_result)
        return cur_min


    def min(self, env, agent_id, l):
        if time.time() > self.end_time:
            return -1
        if l == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_results = [self.max(child, 1-agent_id, l-1) for child in children]
        min_result = min(children_results)
        return min_result

    def max(self, env, agent_id, l):
        if time.time() > self.end_time:
            return -1
        if l == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_results = [self.min(child, 1-agent_id, l-1) for child in children]
        max_result = max(children_results)
        return max_result

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if len(env.passengers) > 0:
             has_gas_to_0 = self.hasGasToTarget(env, taxi_id, env.passengers[0].position)
            #has_gas_to_0 = self.canReachToPassengerAndRefuel(env, taxi, 0)
        else:
            has_gas_to_0 = 0
        if len(env.passengers) > 1:
            has_gas_to_1 = self.hasGasToTarget(env, taxi_id, env.passengers[1].position)
            #has_gas_to_1 = self.canReachToPassengerAndRefuel(env, taxi, 1)
        else:
            has_gas_to_1 = 0
        if taxi.passenger != None:
            target_dest = taxi.passenger.destination
            has_gas_to_2 = self.hasGasToTarget(env, taxi_id, target_dest)
            profit = manhattan_distance(taxi.passenger.position, taxi.passenger.destination)
        else:
            target_dest = taxi.position
            has_gas_to_2 = 0
            profit = 0
        should_refuel = self.shouldRefuel(env, taxi_id, max(self.calculateProfit(env, taxi_id, 0), self.calculateProfit(env, taxi_id, 1), 0), has_gas_to_0, has_gas_to_1, has_gas_to_2)         
        return max(has_gas_to_0 * self.calculateProfit(env, taxi_id, 0), has_gas_to_1 * self.calculateProfit(env, taxi_id, 1)) + has_gas_to_2 * (taxi.fuel - manhattan_distance(taxi.position, target_dest)) * profit + (taxi.fuel - manhattan_distance(taxi.position, env.gas_stations[self.getClosestGasStation(taxi, env)].position)) * should_refuel + taxi.cash



class AgentAlphaBeta(AgentMinimax):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.end_time = time.time() + 0.8*time_limit
        self.alpha = -float("inf")
        step = self.id_minimax(env, agent_id, time_limit)
        return step
    
    def id_minimax(self, env, agent_id, time_limit):
        depth = 1
        while True:
            last_chosen_step = self.minimax(env, agent_id, depth, -float("inf"), float("inf"))
            if time.time() > self.end_time:
                print("max depth: ", depth)
                break 
            else:
                step = last_chosen_step
            depth += 1
        return step

    def minimax(self, env, agent_id, l, alpha, beta):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_results = [self.min(child, 1-agent_id, l-1, alpha, beta) for child in children]
        max_result = max(children_results)
        index_selected = children_results.index(max_result)
        return operators[index_selected]  
    
    # TODO apply pruning , find out what to do with alpha and beta and when to return -inf/+inf, 
    # - should alpha and beta be passed on with the recursion or be a class variable? 
    # - i recomend watching tutorial 6 to go over it. 
    # - notice it inherits from minimax agent so it overrides the function calls
    # - good luck omer i believe in you
    def min(self, env, agent_id, l, alpha, beta):
        if time.time() > self.end_time:
            return -1
        if l == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        # this line might need to change in order to apply pruning
        children_results = [self.max(child, 1-agent_id, l-1) for child in children]
        min_result = min(children_results)
        self.beta = min(self.beta, min_result)
        return min_result

    def max(self, env, agent_id, l):
        if time.time() > self.end_time:
            return -1
        if l == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        # this line might need to change in order to apply pruning
        children_results = [self.min(child, 1-agent_id, l-1) for child in children]
        max_result = max(children_results)
        self.alpha = max(self.alpha, max_result)
        return max_result
        


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
