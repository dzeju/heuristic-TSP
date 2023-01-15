import math
import random
from functools import partial
from wrapt_timeout_decorator import *
import numpy


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def if_then_else(condition, out1, out2):
    return out1() if condition else out2()


def only_if(condition, out):
    def nothing():
        pass
    return out() if condition else nothing()

# eliminacja krzyrzówek
# proste test casy
# semantic backpropagation
# efekt bandlina
# lamarka


class TravelingSalesman(object):
    def __init__(self, cities):
        self.cities = cities
        self.start_city = cities[0]
        self.path = [self.start_city]
        self.remaining_cities = [
            city for city in cities if city != self.start_city]
        self.picked_city = None
        self.second_picked_city = None
        self.total_distance = float('inf')
        self.centroid = (sum([city[0] for city in cities]) / len(cities),
                         sum([city[1] for city in cities]) / len(cities))
        self.penalties = 0
        self.number_of_nodes = len(self.cities)

    def insert_solution(self, solution):
        self.path = solution
        self.remaining_cities = []
        self.calculate_total_distance()
        self.penalties = 0

    def append_picked_city(self):
        if self.picked_city is not None:
            self.path.append(self.picked_city)
            self.picked_city = None
        else:
            self.penalties += 1

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
            self.picked_city = self.remaining_cities.pop(
                self.remaining_cities.index(city))

    def distance_to_starting_city(self, city):
        return distance(city, self.start_city)

    def calculate_given_path_distance(self, path):
        total_distance = 0
        for i in range(len(path)-1):
            total_distance += distance(path[i], path[i+1])
        total_distance += self.distance_to_starting_city(path[-1])
        return total_distance

    def calculate_total_distance(self):
        # Calculate the total distance of the path
        self.total_distance = self.calculate_given_path_distance(self.path)

    def reset(self):
        self.start_city = self.cities[0]
        self.path = [self.start_city]
        self.remaining_cities = [
            city for city in self.cities if city != self.start_city]
        self.picked_city = None
        self.total_distance = float('inf')
        self.centroid = (sum([city[0] for city in self.cities]) / len(self.cities),
                         sum([city[1] for city in self.cities]) / len(self.cities))
        self.penalties = 0
        self.number_of_nodes = len(self.cities)

    @timeout(1, timeout_exception=TimeoutError)
    def run(self, func):
        self.reset()
        func()

        if len(self.path) != self.number_of_nodes:
            self.total_distance = 200000
        else:
            self.calculate_total_distance()

    def run_specific_case(self, func, cities_coords):
        self.cities = cities_coords
        self.run(func)

        if len(self.path) != self.number_of_nodes:
            self.total_distance = 200000
        else:
            self.calculate_total_distance()

    def run_multiple_cases(self, func, cities_coords, solutions):
        overall_total_distance = 0
        overall_penalties = 0
        for i, cities in enumerate(cities_coords):
            self.cities = cities
            self.run(func)
            overall_total_distance += self.total_distance / solutions[i]
            overall_penalties += self.penalties
        return overall_total_distance, overall_penalties

    def distance_from_current_node(self, city):
        return distance(self.path[-1], city)

    def distance_from_centroid(self, city):
        return distance(city, self.centroid)

    def if_centroid_farther_than_last_node(self, out1, out2):
        # self.find_nearest_neighbor_to_current_node()
        def delegate():
            if self.picked_city is None:
                self.penalties += 1
                return
            partial(if_then_else, self.distance_from_centroid(self.picked_city) >
                    self.distance_from_current_node(self.picked_city), out1, out2)
        return delegate

    def if_starting_city_closer_than_last_node(self, out1, out2):
        # self.find_nearest_neighbor_to_current_node()
        def delegate():
            if self.picked_city is None:
                self.penalties += 1
                return
            partial(if_then_else, self.distance_to_starting_city(self.picked_city) <
                    self.distance_from_current_node(self.picked_city), out1, out2)
        return delegate

    def if_any_remaining_cities(self, out):
        return partial(if_then_else, len(self.remaining_cities) > 0, out, self.do_nothing)

    def if_city_already_picked(self, out1, out2):
        return partial(if_then_else, self.picked_city is not None, out1, out2)

    def if_half_remaining_cities(self, out1, out2):
        def delegate():
            partial(if_then_else, len(self.remaining_cities)
                    > len(self.cities)/2, out1, out2)
        return delegate

    def if_second_picked_city_farther_from_centroid(self, out1):
        def delegate():
            if self.picked_city is None or self.second_picked_city is None:
                return
            isFarther = self.distance_from_centroid(self.picked_city) < self.distance_from_centroid(
                self.second_picked_city)
            only_if(isFarther, out1)

        return delegate

    def for_every_remaining_city(self, out):
        def delegate():
            length = len(self.remaining_cities)
            if length == 0:
                self.penalties += 1
                return
            for i in range(length):
                out()
        return delegate

    def find_nearest_neighbor(self, city):
        # Find the nearest neighbor to city
        nearest_neighbor = None
        second_nearest_neighbor = None
        nearest_distance = float('inf')
        length = len(self.remaining_cities)
        if length == 0:
            self.penalties += 1
        for i in range(length):
            curr_distance = distance(city, self.remaining_cities[i])
            if curr_distance < nearest_distance:
                nearest_distance = curr_distance
                second_nearest_neighbor = nearest_neighbor
                nearest_neighbor = self.remaining_cities[i]
        # return nearest_neighbor
        return nearest_neighbor, second_nearest_neighbor

    def find_nearest_city_to_centroid(self):
        # Find the nearest city to the centroid
        self.pick_exact_city(self.find_nearest_neighbor(self.centroid))

    def find_nearest_neighbor_to_current_node(self):
        # Find the nearest neighbor to the current node
        nearest_neighbor, self.second_picked_city = self.find_nearest_neighbor(
            self.path[-1])

        self.pick_exact_city(nearest_neighbor)
        # self.pick_random_city()

    def find_furthest_neighbor_to_current_node(self):
        # Find the furthest neighbor to the current node
        furthest_neighbor = None
        furthest_distance = 0
        length = len(self.remaining_cities)
        if length == 0:
            self.penalties += 1
        for i in range(length):
            curr_distance = distance(self.path[-1], self.remaining_cities[i])
            if curr_distance > furthest_distance:
                furthest_distance = curr_distance
                furthest_neighbor = self.remaining_cities[i]
        self.pick_exact_city(furthest_neighbor)

    def if_two_change(self):
        # from 2-opt
        city1 = random.randint(0, len(self.path) - 1)
        city2 = random.randint(0, len(self.path) - 1)
        path_copy = self.path.copy()

        try:
            path_copy[city1], path_copy[city2] = path_copy[city2], path_copy[city1]
            if (self.calculate_given_path_distance(path_copy) < self.calculate_given_path_distance(self.path)):
                self.path = path_copy
        except IndexError:
            self.penalties += 1
            pass

    def pick_second_nearest_neighbor(self):
        self.pick_exact_city(self.second_picked_city)

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
            nearest_neighbor, _ = self.find_nearest_neighbor(self.path[0])
            self.path.append(nearest_neighbor)
            self.remaining_cities.remove(nearest_neighbor)
            # find nearest neighbor for every remaining city
            while len(self.remaining_cities) > 0:
                nearest_neighbor, _ = self.find_nearest_neighbor(self.path[-1])
                self.path.append(nearest_neighbor)
                self.remaining_cities.remove(nearest_neighbor)
            # add distance from last city to starting city
        except ValueError:
            print("Error in full nearest neighbor algorithm")
            print(self.path)
            print(self.remaining_cities)
            print(self.start_city)

    def strip_heuristic(self):
        try:
            # 1. Divide the set of cities into vertical strips of equal width.
            # 2. Iterate over the strips in order.
            # 3. For each odd-index strip, visit the cities in non-decreasing order of ordinates.
            # 4. For each even-index strip, visit the cities in non-increasing order of ordinates.
            # 5. Repeat steps 3 and 4 until all cities have been visited.

            nearest_neighbor, _ = self.find_nearest_neighbor(self.path[0])
            self.path.append(nearest_neighbor)
            self.remaining_cities.remove(nearest_neighbor)
            # find nearest neighbor for every remaining city
            while len(self.remaining_cities) > 0:
                nearest_neighbor, _ = self.find_nearest_neighbor(self.path[-1])
                self.pick_exact_city(nearest_neighbor)
                partial(if_then_else, self.distance_to_starting_city(self.picked_city) <
                        self.distance_from_current_node(self.picked_city), self.insert_picked_city, self.append_picked_city)()

            # self.path = []
            # width = int(len(self.cities) / 10)
            # print(width)
            # num_strips = int(max(numpy.array(self.cities)[:, 0]) / width) + 1
            # print(num_strips)
            # strips = [[] for _ in range(num_strips)]
            # for city in self.cities:
            #     strip_index = int(city[0] / width)
            #     strips[strip_index].append(city)

            # for strip in strips:
            #     strip.sort(key=lambda x: x[1])

            # for i, strip in enumerate(strips):
            #     # Visit the cities in non-decreasing order of ordinates for odd-index strips
            #     # and in non-increasing order of ordinates for even-index strips
            #     if i % 2 == 0:
            #         order = 1
            #     else:
            #         order = -1
            #     for city in strip[::order]:
            #         if city not in self.path:
            #             self.path.append(city)

            # print(len(self.path), len(self.cities))

        except ValueError:
            print("Error in strip heuristic algorithm")
            print(self.path)
            print(self.remaining_cities)
            print(self.start_city)
