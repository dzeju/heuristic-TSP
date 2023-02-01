import random
from functools import partial
from bitmap_points import BitMapPoints
import copy
import multiprocessing
import os
from contextlib import contextmanager
import tsplib95
from wrapt_timeout_decorator import *
import datetime

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import operator

from TravelingSalesman import TravelingSalesman, test
from drawing import draw_fitness_curve, draw_tree, draw_path, draw_and_display, draw_just_cities

# -----------------------------
# file_names = [
#     'a280',
#     'att48',
#     'berlin52',
#     'gr202',
#     'kroA100',
#     'eil101',
# ]

# problems = []
# for file_name in file_names:
#     problems.append(tsplib95.load('ALL_tsp/' + file_name + '.tsp'))

# solutions = []
# for i, problem in enumerate(problems):
#     solution = tsplib95.load('ALL_tsp/' + file_names[i] + '.opt.tour')
#     solutions.append(problem.trace_tours(solution.tours)[0])

# solutions_paths = []
# for i, problem in enumerate(problems):
#     problem_path = list(tsplib95.load(
#         'ALL_tsp/' + file_names[i] + '.tsp').as_name_dict()['node_coords'].values())
#     solution_indexes = solution = tsplib95.load(
#         'ALL_tsp/' + file_names[i] + '.opt.tour').tours[0]
#     solutions_path = []
#     for index in solution_indexes:
#         solutions_path.append(problem_path[index - 1])
#     solutions_paths.append(solutions_path)


# cities_coord = list()
# for problem in problems:
#     cities_coord.append(list(problem.as_name_dict()['node_coords'].values()))

# -----------------------------
image_points = [
    # BitMapPoints('images/one.png'),
    BitMapPoints('images/two.png'),
    BitMapPoints('images/three.png'),
    # BitMapPoints('images/tests.png')
]
cities_coord = [
    image_points[0].convert_to_list_of_points(),
    image_points[1].convert_to_list_of_points(),
    # image_points[2].convert_to_list_of_points()
]
# solutions = [1, 1]
solutions = [
    # 129.86191021194034,
    289.36373183514655,
    271.9632426076695
    # , 320.1233949157353
]
solutions_paths = cities_coord
# -----------------------------

# print(cities_coord[0])

ts = TravelingSalesman(cities_coord[0])


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
p_set.addPrimitive(ts.for_every_remaining_city, 1,
                   name="FOR_REM_CITIES")
p_set.addPrimitive(ts.for_every_city_in_path, 1,
                   name="FOR_CITIES_IN_PATH")
p_set.addPrimitive(ts.if_starting_city_closer_than_last_node, 2, name="IF_SC")
p_set.addPrimitive(ts.if_centroid_farther_than_last_node, 2, name="IF_CF")
p_set.addPrimitive(ts.if_half_remaining_cities, 2, name="IF_HRC")
p_set.addPrimitive(
    ts.if_second_picked_city_farther_from_centroid, 1, name="IF_SPCFC")
p_set.addPrimitive(
    ts.if_picked_city_crosses_path, 2, name="IF_PCCP")
p_set.addTerminal(ts.append_picked_city, name="T_append")
p_set.addTerminal(ts.insert_picked_city, name="T_insert")
p_set.addTerminal(ts.find_nearest_neighbor_to_current_node, name="T_F_NN")
p_set.addTerminal(ts.find_furthest_neighbor_to_current_node, name="T_F_FN")
p_set.addTerminal(ts.find_nearest_city_to_centroid, name="T_F_CENT")
p_set.addTerminal(ts.pick_second_nearest_neighbor, name="T_P_SNN")
# p_set.addTerminal(ts.if_two_change, name="T_IF_SWAP")
p_set.addTerminal(ts.swap_cities_if_crossing_path, name="T_SWAP")

p_set.addPrimitive(prog2, 2)
p_set.addPrimitive(prog3, 3)
p_set.addPrimitive(prog4, 4)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=p_set, min_=1, max_=8)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=p_set)


def eval_traveling_distance(individual):
    try:
        func = toolbox.compile(individual, pset=p_set)

        # return ts.total_distance,                         # <- this is the original
        # return (ts.total_distance + ts.penalties * 10),   # <- this is for single case
        total_distance, penalties = ts.run_multiple_cases(
            func, cities_coord, solutions)  # <- this is for multiple cases
        return (total_distance + penalties * 10),
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


toolbox.register("evaluate", eval_traveling_distance)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=p_set)
# toolbox.register("mutate", gp.mutNodeReplacement, pset=p_set)
toolbox.decorate("mate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=10))


def save_logs_and_drawings(best_indiv, cities_coord, func, mutpb, cxpb, ngen, log):
    drawings_list = []

    all_results_list = []

    for i, cities in enumerate(cities_coord):
        ts.run_specific_case(func, cities)
        best_indiv_copy = copy.deepcopy(ts)

        ts.run_specific_case(ts.nearest_neighbor_heuristic, cities)
        nearest_neighbor_copy = copy.deepcopy(ts)

        ts.run_specific_case(ts.strip_heuristic, cities)
        strip_copy = copy.deepcopy(ts)

        # ts.run_specific_case(ts.christofides_heuristic, cities)
        # christofides_copy = copy.deepcopy(ts)

        ts.insert_solution(solutions_paths[i])
        solution_copy = copy.deepcopy(ts)

        # name = problems[i].as_name_dict()['name']
        name = str(i)

        result_list = ['\n',
                       name,
                       'evolution path length:        ' +
                       str(best_indiv_copy.total_distance),
                       'nearest neighbor path length: ' +
                       str(nearest_neighbor_copy.total_distance),
                       'strip path length:            ' +
                       str(strip_copy.total_distance),
                       #    'christofides path length:     ' +
                       #    str(christofides_copy.total_distance),
                       'optimal path length:          ' +
                       str(solution_copy.total_distance)]

        for j, result in enumerate(result_list):
            print(result)
            all_results_list.append(result)
            # if (save):
            #     file.writelines([result + '\n'])

        drawings = {'evolution': best_indiv_copy.path,
                    'nearest_neighbor': nearest_neighbor_copy.path,
                    'strip': strip_copy.path,
                    # 'christofides': christofides_copy.path,
                    'optimal': solution_copy.path}

        drawings_list.append(drawings)

    toSave = input('Save results? (y/n): ')
    if toSave == 'y' or toSave == 'Y' or toSave == '':
        date_time = str(datetime.datetime.now())
        MAIN_DIR = os.path.join('results', date_time)
        os.mkdir(MAIN_DIR)

        file = open('results/'+date_time+'/logs.txt', 'w')

        file.writelines([date_time])
        file.writelines(['\n\nmutpb: ' + str(mutpb), '\n', 'cxpb: ' +
                        str(cxpb), '\n', 'ngen: ' + str(ngen)])
        file.writelines(['\n\nbest individual: ', str(best_indiv)])
        file.writelines([
                        # '\n\nexpr_mut: ', str(toolbox.__getattribute__('expr_mut')),
                        '\nselect: ', str(toolbox.__getattribute__('select')),
                        '\nmate: ', str(toolbox.__getattribute__('mate')),
                        '\nmutate: ', str(toolbox.__getattribute__('mutate'))])

        file.writelines(['\n\n', 'results:'])
        for i, result in enumerate(all_results_list):
            file.writelines([result + '\n'])

        file.close()

        nodes, edges, labels = gp.graph(best_indiv)
        draw_tree(nodes, edges, labels, date_time)

        draw_fitness_curve(log.chapters["fitness"].select("min"), date_time)

        for i, drawings in enumerate(drawings_list):
            # name = problems[i].as_name_dict()['name']
            name = str(i)
            draw_path(drawings, cities_coord[i], name, date_time)


def main():
    random.seed(318)
    cxpb, mutpb, ngen, npop = 0.5, 0.2, 50, 1000

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=mstats,
                                   halloffame=hof, verbose=True)

    best_indiv = hof[0]
    func = toolbox.compile(best_indiv, pset=p_set)

    save_logs_and_drawings(best_indiv, cities_coord,
                           func, mutpb, cxpb, ngen, log)


def check_solution_algorithm():
    try:
        individual = input('Individual: ')
        func = toolbox.compile(individual, pset=p_set)
    except SyntaxError as e:
        print('Wrong individual: ', e)
        return

    for i, cities in enumerate(cities_coord):
        ts.run_specific_case(func, cities)
        best_indiv_copy = copy.deepcopy(ts)

        ts.run_specific_case(ts.nearest_neighbor_heuristic, cities)
        nearest_neighbor_copy = copy.deepcopy(ts)

        ts.run_specific_case(ts.strip_heuristic, cities)
        strip_copy = copy.deepcopy(ts)

        # ts.run_specific_case(ts.christofides_heuristic, cities)
        # christofides_copy = copy.deepcopy(ts)

        ts.insert_solution(solutions_paths[i])
        solution_copy = copy.deepcopy(ts)

        print('\nevolution path length:        ',
              best_indiv_copy.total_distance)
        print('nearest neighbor path length: ',
              nearest_neighbor_copy.total_distance)
        print('strip path length:            ', strip_copy.total_distance)
        # print('christofides path length:     ',
        #       christofides_copy.total_distance)
        print('optimal path length:          ', solution_copy.total_distance)

        draw_and_display(best_indiv_copy.path, cities)


if __name__ == '__main__':
    print('1. run evolution algorithm',
          '2. check solution algorithm',
          '3. draw just cites',
          sep='\n')

    choice = input('Choice: ')

    if choice == '1':
        with multiprocessing.Pool() as pool:
            # pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)
            main()
            # pool.close()
    elif choice == '2':
        check_solution_algorithm()
    elif choice == '3':
        for i, cities in enumerate(cities_coord):
            # ts = TravelingSalesman(cities)
            # print(ts.number_of_nodes)
            # draw_just_cities(cities, file_names[i], "JUST CITIES")
            draw_just_cities(cities, str(i+1), "JUST CITIES")
