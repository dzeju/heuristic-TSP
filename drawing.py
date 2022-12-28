import matplotlib.pyplot as plt
import networkx as nx
import numpy
from networkx.drawing.nx_agraph import graphviz_layout


def draw_fitness_curve(fitness_curve):
    plt.plot(fitness_curve)
    plt.title('fitness curve')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()


def draw_path(paths_dict, cities_coord):
    # create as many subplots as items in paths list
    paths = list(paths_dict.values())
    names = list(paths_dict.keys())

    length = int(len(paths) / 2) + len(paths) % 2

    fig, ax = plt.subplots(length, length)
    cities_coord2 = numpy.array(cities_coord)

    # Iterate through the list of items
    for i, item in enumerate(paths):
        # Create a subplot for each item
        xy = numpy.array(item)
        ax[i % length][int(i / length)].plot(xy[:, 0], xy[:, 1], 'ro-')
        ax[i % length][int(i / length)].plot(cities_coord2[:, 0], cities_coord2[:, 1], 'bo')
        ax[i % length][int(i / length)].set_title(names[i])

    # xy = numpy.array(paths)
    # for path in xy:
    #     plt.plot(path[:, 0], path[:, 1])
    # # plt.plot(xy[:, 0], xy[:, 1], 'ro-')
    # plt.plot(cities_coord2[:, 0], cities_coord2[:, 1], 'bo')
    plt.show()


def draw_tree(nodes, edges, labels):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")
    # pos = nx.spring_layout(g)
    # pos = nx.planar_layout(g, scale=2)
    # pos = nx.circular_layout(g)
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=6)
    plt.show()
