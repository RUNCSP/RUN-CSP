from model import RUN_CSP
from train import train
from csp_utils import CSP_Instance, mc_weighted_language

import data_utils
import argparse
import os
import numpy as np
import glob
import networkx as nx
import csv
import numpy as np

from tqdm import tqdm


def get_random_graph():
    graph = nx.gnm_random_graph(100, np.random.randint(100, 300))
    weights = {e: np.random.choice([1, -1]) for e in graph.edges()}
    nx.set_edge_attributes(graph, weights, 'weight')
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used during training')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('-m', '--model_dir', type=str, help='The model directory of a trained network')
    parser.add_argument('-t', '--t_max', type=int, default=100,
                        help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None,
                        help='A path to a training set of graphs in the NetworkX adj_list format. If left unspecified, random instances are used.')
    parser.add_argument('-v', '--n_variables', type=int, default=100,
                        help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('--degree', type=int, default=3,
                        help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=400,
                        help='Number of instances for training. Only used when --data_path is not specified.')
    parser.add_argument('-s', '--save_path', type=str, help='Path to a csv file to store results')
    args = parser.parse_args()

    language = mc_weighted_language

    if args.data_path is not None:
        print('loading graphs...')
        # graphs = [data_utils.load_col_graph(p) for p in tqdm(glob.glob(args.data_path))]
        # names = [os.path.basename(p) for p in glob.glob(args.data_path)]
        # instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ', name=name) for g, name in zip(graphs, names)]
        path = os.path.join('data', args.data_path, '**')
        graphs = data_utils.load_graphs(path)
        instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ') for g in graphs]
    else:
        print(f'Generating {args.n_instances} training instances')
        graphs = [get_random_graph() for _ in range(args.n_instances)]
        instances = [CSP_Instance.graph_to_weighted_mc_instance(g) for g in tqdm(graphs)]

    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)
    net = RUN_CSP(args.model_dir, language=language)
    train(net, train_batches, t_max=args.t_max, epochs=args.epochs)


if __name__ == '__main__':
    main()
