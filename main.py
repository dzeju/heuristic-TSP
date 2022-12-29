import random
from functools import partial
from bitmap_points import BitMapPoints
import copy
import multiprocessing
import signal
from contextlib import contextmanager
import tsplib95
from wrapt_timeout_decorator import *

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from TravelingSalesman import TravelingSalesman
from drawing import draw_fitness_curve, draw_tree, draw_path

# 10 cities coordinates
# cities_coord = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [3, 0]]

# image_points = BitMapPoints('shape2.png')
# cities_coord = image_points.convert_to_list_of_points()
problems = [
            # tsplib95.load('/Users/dzeju/Documents/PycharmProjects/heuristic-TSP/ALL_tsp/bayg29.tsp'),
            # tsplib95.load('/Users/dzeju/Documents/PycharmProjects/heuristic-TSP/ALL_tsp/berlin52.tsp'),
            tsplib95.load('/Users/dzeju/Documents/PycharmProjects/heuristic-TSP/ALL_tsp/bier127.tsp'),
            tsplib95.load('/Users/dzeju/Documents/PycharmProjects/heuristic-TSP/ALL_tsp/burma14.tsp'),
            # tsplib95.load('/Users/dzeju/Documents/PycharmProjects/heuristic-TSP/ALL_tsp/ch130.tsp')
]
cities_coord = list()
for problem in problems:
    cities_coord.append(list(problem.as_name_dict()['node_coords'].values()))

ts = TravelingSalesman(cities_coord[0])

# cities_coord = list(problem.as_name_dict()['display_data'].values())
# print(cities_coord)


# cities_coord = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]
# cities_coord = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [3, 0], [3, 1], [3, 2], [2, 3], [1, 3], [0, 3], [4, 0], [4, 1], [4, 2], [4, 3], [3, 4], [2, 4], [1, 4], [0, 4], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [6, 7], [5, 7], [4, 7], [3, 7], [2, 7], [1, 7], [0, 7], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [7, 8], [6, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [8, 9], [7, 9], [6, 9], [5, 9], [4, 9], [3, 9], [2, 9], [1, 9], [0, 9]]
# cities_coord = numpy.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1], [2, 0]])


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def progn(*args):
    for arg in args:
        arg()


def prog1(out1):
    return partial(progn, out1)


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def prog4(out1, out2, out3, out4):
    return partial(progn, out1, out2, out3, out4)


p_set = gp.PrimitiveSet("MAIN", 0)
# p_set.addPrimitive(ts.full_nearest_neighbor_algorithm, 1)
p_set.addPrimitive(ts.for_every_remaining_city, 1, name="for_every_remaining_city")
# p_set.addPrimitive(ts.pick_exact_city, 1, name="pick_exact_city")
p_set.addPrimitive(ts.if_starting_city_closer_than_last_node, 2, name="IF_SC")
p_set.addPrimitive(ts.if_centroid_farther_than_last_node, 2, name="IF_CF")
# p_set.addPrimitive(ts.if_any_remaining_cities, 1, name="IF_ARC")
# p_set.addPrimitive(ts.if_half_remaining_cities, 2, name="IF_HRC")
# p_set.addTerminal(ts.pick_random_city, name="T_PRC")
p_set.addTerminal(ts.append_picked_city, name="T_append")
p_set.addTerminal(ts.insert_picked_city, name="T_insert")
p_set.addTerminal(ts.find_nearest_neighbor_to_current_node, name="T_find_n_n")

p_set.addPrimitive(prog1, 1)
p_set.addPrimitive(prog2, 2)
p_set.addPrimitive(prog3, 3)
# p_set.addPrimitive(prog4, 4)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=p_set, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=p_set)


def run_individual(func, cities_coords):
    total_distance, penalties = ts.run_multiple_cases(func, cities_coords)
    return total_distance, penalties


def eval_traveling_distance(individual):
    try:
        func = toolbox.compile(individual, pset=p_set)
        total_distance, penalties = run_individual(func, cities_coord)
        # return ts.total_distance,                         # <- this is the original
        # return (ts.total_distance + ts.penalties * 10),   # <- this is for single case
        return (total_distance + penalties * 10),         # <- this is for multiple cases
    except SyntaxError as s_err:
        print(s_err)
        return float('inf'),
    except TimeoutError as t_err:
        # print(t_err)
        return float('inf'),
    except Exception as e:
        print('generic', e)
        # print(individual)
        # nodes, edges, labels = gp.graph(individual)
        # draw_tree(nodes, edges, labels)
        return float('inf'),


# toolbox.register("evaluate", eval_traveling_distance)
toolbox.register("evaluate", eval_traveling_distance)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=p_set)


def main():
    random.seed(318)

    pop = toolbox.population(n=50)
    # print(pop[0])
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 7000, stats=mstats,
                                   halloffame=hof, verbose=True)

    best_indiv = hof[0]
    func = toolbox.compile(best_indiv, pset=p_set)
    ts.run(func)
    best_indiv_copy = copy.deepcopy(ts)

    ts.run(ts.nearest_neighbor_heuristic)
    nearest_neighbor_copy = copy.deepcopy(ts)

    ts.run(ts.strip_heuristic)
    strip_copy = copy.deepcopy(ts)

    print('evolution path length:        ', best_indiv_copy.total_distance)
    print('nearest neighbor path length: ', nearest_neighbor_copy.total_distance)
    print('strip path length:            ', strip_copy.total_distance)

    drawings = {'evolution': best_indiv_copy.path,
                'nearest_neighbour': nearest_neighbor_copy.path,
                'strip': strip_copy.path}

    draw_path(drawings, cities_coord)

    nodes, edges, labels = gp.graph(best_indiv)
    draw_tree(nodes, edges, labels)

    # draw first created tree
    # nodes, edges, labels = gp.graph(pop[0])
    # draw_tree(nodes, edges, labels)

    draw_fitness_curve(log.chapters["fitness"].select("min"))
    best_path = hof[0]
    print(best_path)


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    main()

