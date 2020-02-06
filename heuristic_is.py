import os
import glob
import networkx as nx
from networkx.algorithms.approximation import independent_set
import data_utils
import numpy as np
import argparse
import csv
import collect_optima
from tqdm import tqdm


def greedy(g):
    """
    Greedy heuristic for Max-IS 
    :param g: A networkx graph
    :return: An indepent set of nodes 
    """

    # get neighbours and degree for each node
    neighbours_degrees = [(n, set(nx.neighbors(g, n))) for n in g.nodes()]
    
    mis = set()
    while len(neighbours_degrees) != 0:
        # add node with lowest degree to set
        neighbours_degrees.sort(key=lambda x: len(x[1]))
        node, remove = neighbours_degrees[0]
        mis.add(node)

        # remove the node and its neighbours
        neighbours_degrees = [(n, neigh - remove) for n, neigh in neighbours_degrees[1:] if n not in remove]
    return mis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str,  help='the name of the data set')
    args = parser.parse_args()

    save_path = os.path.join(args.data_path, 'greedy.csv')    
    folders = [p for p in glob.glob(os.path.join(args.data_path, '*')) if os.path.isdir(p)]
    rows = []
    for f in folders:
        graphs = data_utils.load_graphs(f) 
        is_sizes = [len(greedy(g)) for g in tqdm(graphs)]
        mean_size = np.mean(is_sizes)
        rows.append((collect_optima.get_ratio(f), mean_size))
        print(f'Mean size for {f}: {mean_size}')

    rows = sorted(rows, key=lambda x: x[0])
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='unix')
        writer.writerow(['Degree', 'Mean Size'])
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    main()
