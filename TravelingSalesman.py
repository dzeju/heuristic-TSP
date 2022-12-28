import math
import random
from functools import partial


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def if_then_else(condition, out1, out2):
    return out1() if condition else out2()


def only_if(condition, out):
    return out() if condition else None


class TravelingSalesman(object):
    def __init__(self, cities):
        self.cities = cities
        self.start_city = cities[0]
        self.path = [self.start_city]
        self.remaining_cities = [city for city in cities if city != self.start_city]
        self.picked_city = None
        self.total_distance = float('inf')
        self.centroid = (sum([city[0] for city in cities]) / len(cities), sum([city[1] for city in cities]) / len(cities))
        self.penalties = 0
        self.number_of_nodes = len(self.cities)

    def append_picked_city(self):
        # Pick a random city from the remaining cities
        # if self.picked_city in self.path:
        #     raise Exception("City already in path")
        # if self.picked_city in self.remaining_cities:
        #     print('picked city: ', self.picked_city)
        #     print('remaining cities: ', self.remaining_cities)
        #     raise Exception("City still in remaining cities")
        if self.picked_city is not None:
            self.path.append(self.picked_city)
            self.picked_city = None
        else:
            self.penalties += 1
        # if len(self.path) + len(self.remaining_cities) != len(self.cities):
        #     print('path:      ', self.path)
        #     print('remaining: ', self.remaining_cities)
        #     raise Exception("Path and remaining cities do not add up")

    def insert_picked_city(self):
        if self.picked_city is not None:
            self.path.insert(0, self.picked_city)
            self.picked_city = None
        else:
            self.penalties += 1

    def do_nothing(self):
        pass

    def pick_random_city(self):
        if len(self.remaining_cities) > 0:
            city_to_pick = random.choice(self.remaining_cities)
            self.picked_city = city_to_pick
            self.remaining_cities.remove(city_to_pick)

    def pick_next_city(self):
        # Pick next city from the remaining cities
        if len(self.remaining_cities) > 0:
            if self.picked_city is not None:
                self.remaining_cities.append(self.picked_city)
            self.picked_city = self.remaining_cities.pop(0)

    def pick_exact_city(self, city):
        if city == self.picked_city:
            pass
        elif city in self.remaining_cities:
            if self.picked_city is not None:
                self.remaining_cities.append(self.picked_city)
            self.picked_city = self.remaining_cities.pop(self.remaining_cities.index(city))

    def distance_to_starting_city(self, city):
        return distance(city, self.start_city)

    def calculate_total_distance(self):
        # Calculate the total distance of the path
        total_distance = 0
        for i in range(len(self.path)-1):
            total_distance += distance(self.path[i], self.path[i+1])
        total_distance += self.distance_to_starting_city(self.path[-1])
        self.total_distance = total_distance

    def reset(self):
        self.start_city = self.cities[0]
        self.path = [self.start_city]
        self.remaining_cities = [city for city in self.cities if city != self.start_city]
        self.picked_city = None
        self.total_distance = float('inf')
        self.centroid = (sum([city[0] for city in self.cities]) / len(self.cities),
                         sum([city[1] for city in self.cities]) / len(self.cities))
        self.penalties = 0
        self.number_of_nodes = len(self.cities)

    def run(self, func):
        self.reset()
        func()

        if len(self.path) != self.number_of_nodes:
            self.total_distance = 200000
        else:
            self.calculate_total_distance()

    def run_multiple_cases(self, func, cities_coords):
        overall_total_distance = 0
        overall_penalties = 0
        for cities in cities_coords:
            self.cities = cities
            self.run(func)
            overall_total_distance += self.total_distance
            overall_penalties += self.penalties
        return overall_total_distance, overall_penalties

    def distance_from_current_node(self, city):
        return distance(self.path[-1], city)

    def distance_from_centroid(self, city):
        return distance(city, self.centroid)

    def if_centroid_farther_than_last_node(self, out1, out2):
        # self.find_nearest_neighbor_to_current_node()
        if self.picked_city is None:
            self.penalties += 1
            # self.find_nearest_neighbor_to_current_node()
            return self.do_nothing
        return partial(if_then_else, self.distance_from_centroid(self.picked_city) >
                       self.distance_from_current_node(self.picked_city), out1, out2)

    def if_starting_city_closer_than_last_node(self, out1, out2):
        # self.find_nearest_neighbor_to_current_node()
        if self.picked_city is None:
            self.penalties += 1
            # self.find_nearest_neighbor_to_current_node()
            return self.do_nothing
        return partial(if_then_else, self.distance_to_starting_city(self.picked_city) <
                       self.distance_from_current_node(self.picked_city), out1, out2,)

    def if_any_remaining_cities(self, out):
        return partial(if_then_else, len(self.remaining_cities) > 0, out, self.do_nothing)

    def if_city_already_picked(self, out1, out2):
        return partial(if_then_else, self.picked_city is not None, out1, out2)

    def if_half_remaining_cities(self, out1, out2):
        return partial(if_then_else, len(self.remaining_cities) > len(self.cities)/2, out1, out2)

    def for_every_remaining_city(self, out):
        def for_every():
            length = len(self.remaining_cities)
            for i in range(length):
                out()
        return for_every

    def find_nearest_neighbor(self, city):
        # Find the nearest neighbor to city
        nearest_neighbor = None
        nearest_distance = float('inf')
        length = len(self.remaining_cities)
        for i in range(length):
            curr_distance = distance(city, self.remaining_cities[i])
            if curr_distance < nearest_distance:
                nearest_distance = curr_distance
                nearest_neighbor = self.remaining_cities[i]
        # return nearest_neighbor
        return nearest_neighbor

    def find_nearest_neighbor_to_current_node(self):
        # Find the nearest neighbor to the current node
        self.pick_exact_city(self.find_nearest_neighbor(self.path[-1]))
        # self.pick_random_city()

    def remaining_cities(self):
        return len(self.remaining_cities)

    @property
    def current_city(self):
        return self.path[-1]

    @property
    def current_path(self):
        return self.path

    @property
    def total_cities(self) -> int:
        return len(self.cities)

    # basic heuristics
    def nearest_neighbor_heuristic(self):
        try:
            nearest_neighbor = self.find_nearest_neighbor(self.start_city)
            self.path.append(nearest_neighbor)
            self.remaining_cities.remove(nearest_neighbor)
            # find nearest neighbor for every remaining city
            while len(self.remaining_cities) > 0:
                nearest_neighbor = self.find_nearest_neighbor(self.path[-1])
                self.path.append(nearest_neighbor)
                self.remaining_cities.remove(nearest_neighbor)
            # add distance from last city to starting city
        except ValueError:
            print("Error in full nearest neighbor algorithm")
            print(self.path)
            print(self.remaining_cities)
            print(self.start_city)

    def strip_heuristic(self):
        # (Daganzo, 1984)
        # 1. Find the nearest neighbor to the starting city
        # 2. Find the nearest neighbor to the last city
        # 3. If the distance from the starting city to the nearest neighbor is less than the distance
        # from the last city to the nearest neighbor, then the nearest neighbor is the starting city
        # 4. Otherwise, the nearest neighbor is the last city
        # 5. Repeat steps 2-4 until all cities have been visited
        # 6. Add the distance from the last city to the starting city
        try:
            nearest_neighbor = self.find_nearest_neighbor(self.start_city)
            self.path.append(nearest_neighbor)
            self.remaining_cities.remove(nearest_neighbor)
            # find nearest neighbor for every remaining city
            while len(self.remaining_cities) > 0:
                nearest_neighbor = self.find_nearest_neighbor(self.path[-1])
                self.pick_exact_city(nearest_neighbor)
                self.if_starting_city_closer_than_last_node(self.insert_picked_city, self.append_picked_city)()
        except ValueError:
            print("Error in strip heuristic algorithm")
            print(self.path)
            print(self.remaining_cities)
            print(self.start_city)





