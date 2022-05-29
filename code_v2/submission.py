from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random


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
        #Calculate PROFIT for each passenger.
        taxi = env.get_taxi(taxi_id)
        # if taxi.passenger != None:
        #     return taxi.fuel - manhattan_distance(taxi.position, taxi.passenger.destination) + taxi.cash * 2
        # if len(env.passengers) > 0:
        #     best_passenger = self.getBestPassenger(taxi, env, taxi_id)
        # if manhattan_distance(taxi.position, env.passengers[best_passenger].destination) > taxi.fuel:
        #     return taxi.fuel - manhattan_distance(taxi.position, env.gas_stations[self.getClosestGasStation(taxi, env, taxi_id)].position) + taxi.cash
        # #TODO: Add consideration to fuel.
        # return self.calculateProfit(env, taxi_id, 0) +  self.calculateProfit(env, taxi_id, 1) + taxi.cash
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
        # return max(has_gas_to_0 * self.calculateProfit(env, taxi_id, 0), has_gas_to_1 * self.calculateProfit(env, taxi_id, 1)) + has_gas_to_2 * (taxi.fuel - manhattan_distance(taxi.position, target_dest)) * 1.5 + (taxi.fuel - manhattan_distance(taxi.position, env.gas_stations[self.getClosestGasStation(taxi, env)].position)) * should_refuel + taxi.cash


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
# Things to consider:
# 2. PROFIT.
# 4. Fuel remaining.
# 5. Distance to gas stations.
# 6. Cash available.



# 1. Find passenger with max profit.
# 2. Calculate distance to it. === D



# P === Profit = (distance from passenger to dest) - (distance from taxi to passenger)
# F === Fuel remaining = current fuel - ((distance from passenger to dest) + (distance from taxi to passenger))



class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        next_state, score = self.id_minimax(env, agent_id, time_limit)
        return 

    def id_minimax(self, env, agent_id, time_limit):
        start_time = 
        


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
