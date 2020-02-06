from model import RUN_CSP
from csp_utils import CSP_Instance

import numpy as np
import argparse

import time

from tqdm import tqdm
import csv


def evaluate_and_save(path, network, eval_instances, t_max, attempts=64):

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='unix')
        head = ['file', 'best cost', 'best ratio'] + [i for i in range(1, attempts + 1)]
        writer.writerow(head)

        conflict_ratios = []
        for i, instance in tqdm(enumerate(eval_instances)):
            output_dict = network.predict_boosted(instance, iterations=t_max, attempts=attempts)
            assignment = output_dict['assignment']

            conflicts = instance.count_conflicts(assignment)
            conflict_ratio = conflicts / instance.n_clauses
            conflict_ratios.append(conflict_ratio)

            print(f'Conflicts for instance {i if instance.name is None else instance.name}: {conflicts}, Valid {instance.n_clauses - conflicts}')

            line = [instance.name, conflicts, conflict_ratio] + output_dict['all_conflicts']
            writer.writerow(line)

        mean_conflict_ratio = np.mean(conflict_ratios)
        print(f'mean conflict ratio for evaluation instances: {mean_conflict_ratio}')
        return conflict_ratios


def evaluate_boosted(network, eval_instances, t_max, attempts=64):
    """
    Evaluate RUN-CSP Network with boosted predictions
    :param network: A RUN_CSP network
    :param eval_instances: A list of CSP instances for evaluation
    :param t_max: Number of RUN_CSP iterations on each instance
    :param attempts: Number of parallel attempts for each instance
    """

    conflict_ratios = []
    for i, instance in enumerate(eval_instances):

        #start = time.time()
        output_dict = network.predict_boosted(instance, iterations=t_max, attempts=attempts)
        #end = time.time()
        #print(f'Total Time: {end - start}s')

        assignment = output_dict['assignment']
        
        conflicts = instance.count_conflicts(assignment)
        conflict_ratio = conflicts / instance.n_clauses
        conflict_ratios.append(conflict_ratio)

        print(f'Conflicts for instance {i if instance.name is None else instance.name}: {conflicts}, Valid {instance.n_clauses - conflicts}')

    mean_conflict_ratio = np.mean(conflict_ratios)
    print(f'mean conflict ratio for evaluation instances: {mean_conflict_ratio}')
    return conflict_ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the trained RUN-CSP instance')
    parser.add_argument('-v', '--n_variables', type=int, default=100, help='Number of variables in each training instance.')
    parser.add_argument('-c', '--n_clauses', type=int, default=100, help='Number of clauses in each training instance.')
    parser.add_argument('-i', '--n_instances', type=int, default=1000, help='Number of instances for training.')
    parser.add_argument('-t', '--t_max', type=int, default=40, help='Number of network iterations t_max')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Number of attempts to boost results')
    args = parser.parse_args()

    # create RUN_CSP instance for given constraint language
    network = RUN_CSP.load(args.model_dir)
    language = network.language

    print(f'Generating {args.n_instances} evaluation instances')
    eval_instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, language, args.weighted) for _ in tqdm(range(args.n_instances))]

    # train and store the network
    evaluate_boosted(network, eval_instances, args.t_max, args.attempts)


if __name__ == '__main__':
    main()
