import os
import random
import argparse
import time
import collections
import numpy as np
import networkx as nx
from tqdm import tqdm # progress bars
import pycosat # sat solver
import data_utils # my stuff
# import sat_coloring

time_for_positive_instances = 0.0
positive_counter = 0
time_for_negative_instances = 0.0
negative_counter = 0

# define command line parser
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graph_count', type=int, help='number of graphs to generate')
parser.add_argument('-v', '--vertex_count', type=int, help='number of nodes in each graph')
parser.add_argument('-n', '--name', help='name of the generated data set (subfolder of data)')
parser.add_argument('-k', '--parts', type=int, help='generates 50 percent k partite graphs')
parser.add_argument('-r', '--random_edges_factor', default=2, type=int, help='random_edges_factor * v many edges will be added as initialization')
parser.add_argument('-b', '--block_size', type=int, default=1000, help='Number of graphs in each block')
parser.add_argument('--lemos', help='generate a lemos style dataset', action='store_true')

def compute_basic_clauses(no_of_nodes, no_of_colors, variables=None):
    '''returns the basic clauses which are identical for every instance of size k as a list'''
    if variables is None:
        variables = compute_variables(no_of_nodes,no_of_colors)
    N = no_of_nodes
    k = no_of_colors
    # init variables
    variables = np.reshape(range(1, k * N + 1), [N, k])
    
    some_color = [list(variables[n]) for n in range(N)]
    one_color = [[-variables[n, i], -variables[n, j]] for n, i, j in filter(lambda x: x[1] != x[2], np.ndindex((N, k, k)))]
    
    fixed_clauses = some_color + one_color
    #convert all numbers to python's integer
    fixed_clauses = [[int(literal) for literal in clause] for clause in fixed_clauses]
    
    return fixed_clauses


def compute_variables(no_of_nodes, no_of_colors):
    '''helper function to compute the variables used in the SAT instance'''
    return np.reshape(range(1, no_of_colors * no_of_nodes + 1), [no_of_nodes, no_of_colors])


def compute_edge_clauses(graph,no_of_colors, variables=None):
    '''returns the edge constraints for a given graph'''
    if variables is None:
        variables = compute_variables(len(graph.nodes()), no_of_colors)
    edge_constraints = []
    for (u, v) in graph.edges():
        constraints = [[int(-variables[int(u), i]), int(-variables[int(v), i])] for i in range(no_of_colors)]
        edge_constraints += constraints
    return edge_constraints
     
    
def compute_single_edge_constraint(edge, no_of_colors, variables):
    '''adding a single edge to a graph adds exactly those constraints to the sat instance'''
    return [[int(-variables[edge[0], i]), int(-variables[edge[1], i])] for i in range(no_of_colors)]


def find_coloring(graph, no_of_colors, fixed_clauses=None, variables=None, edge_constraints=None):
    '''mainly a wrapper around the pycosat solver, gets the graph and some precomputed stuff to compute the solution'''
    
    # used for speedup, large parts of the SAT instance can be reused
    no_of_nodes = len(graph.nodes())
    if variables is None:
        variables = compute_variables(no_of_nodes, no_of_colors)
    if fixed_clauses is None: 
        fixed_clauses = compute_basic_clauses(no_of_nodes, no_of_colors, variables=variables) 
    if edge_constraints is None:
        edge_constraints = compute_edge_clauses(graph, no_of_colors, variables=variables)
        
    # compute solution for SAT instance
    cnf = fixed_clauses + edge_constraints
    start_time = time.perf_counter()
    solution = pycosat.solve(cnf)
    elapsed_time = time.perf_counter() - start_time

    if solution == 'UNSAT' or solution == 'UNKNOWN':
        global time_for_negative_instances
        global negative_counter
        time_for_negative_instances += elapsed_time
        negative_counter += 1
        return False, None
    else:
        global time_for_positive_instances
        global positive_counter
        time_for_positive_instances += elapsed_time
        positive_counter += 1
        solution = np.reshape(solution, [no_of_nodes, no_of_colors])
        solution = np.argmax(solution, axis=1)
        return True, solution


def generate(no_of_nodes, random_edges_factor, no_of_colors=3, edges_per_sat_check=2):
    ''' generates two graphs with n nodes that are colorable with no_of_colors many colors'''
    
    # generate empty graph
    nodes = list(range(no_of_nodes))
    graph = None
    
    # variables and basic clauses for faster sat solving
    variables = compute_variables(no_of_nodes, no_of_colors)
    basic_clauses = compute_basic_clauses(no_of_nodes,no_of_colors, variables=variables)
    edge_constraints = None

    sat = False 
    solution = None
    # initialize a graph with a number of random edges in the beginning. Makes the graph use all its nodes. 
    # (otherwise we would generate graphs with 1000 nodes and 100 edges)
    no_of_random_edges = int(random_edges_factor * no_of_nodes)
    while not sat:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        # start with some random edges
        nodes_from = random.choices(nodes,k=no_of_random_edges)
        nodes_to = random.choices(nodes,k=no_of_random_edges)
        pairs = zip(nodes_from,nodes_to)
        graph.add_edges_from(pairs) # duplicate edges are ignored
        graph.remove_edges_from(graph.selfloop_edges()) # remove any selfloop edges that might have been inserted
        
        edge_constraints = compute_edge_clauses(graph, no_of_colors, variables=variables)
        sat,solution = find_coloring(graph,no_of_colors,fixed_clauses=basic_clauses, 
                                     variables=variables, edge_constraints=edge_constraints)
        # discount the number of edges for the case of an unsatisfiable instance to start over
        no_of_random_edges = int((no_of_random_edges/4)*3)
        if not sat: 
            print("oh no!") # should happen rarely as long as random_edges_factor < 4

    # add edges in groups of edges_per_sat_check
    last_set_of_edges = []
    while sat:
        counted_colors = collections.Counter(random.choices(range(no_of_colors),k=edges_per_sat_check))
        last_set_of_edges = []
        for color in counted_colors:
            nodes_with_color = list(np.where(solution == color)[0])
            if len(nodes_with_color) >= 2:
                for _ in range(counted_colors[color]):
                    last_set_of_edges.append(random.sample(nodes_with_color,2))
        
        # actually add those edges to the graph
        graph.add_edges_from(last_set_of_edges)
        for edge in last_set_of_edges:
            edge_constraints += compute_single_edge_constraint(edge, no_of_colors, variables)
        
        sat, solution = find_coloring(graph, no_of_colors,fixed_clauses=basic_clauses, 
                                      variables=variables, edge_constraints=edge_constraints)
        
    # now we have an unsatisfiable graph and the last (checked) satisfiable graph differs by the edges in last_set_of_edges
    # we add the edges one by one to stop when inserting one that makes the problem unsatisfiable
    final_edge = None
    if edges_per_sat_check == 1:
        final_edge = last_set_of_edges[0] # there is only one, so we don't need to double check anything
    else:
        graph.remove_edges_from(last_set_of_edges)
        edge_constraints = compute_edge_clauses(graph, no_of_colors, variables=variables)

        for index,edge in enumerate(last_set_of_edges):
            graph.add_edge(edge[0],edge[1])
            # we don't need to check the last edge explicitly
            if index < len(last_set_of_edges)-1:
                edge_constraints += compute_single_edge_constraint(edge, no_of_colors, variables)
                sat,_ = find_coloring(graph,no_of_colors,fixed_clauses=basic_clauses, 
                                      variables=variables, edge_constraints=edge_constraints)
            if not sat or index == len(last_set_of_edges)-1:
                final_edge = edge
                break
    
    pos_graph = graph.copy()
    pos_graph.remove_edges_from([final_edge])

    return pos_graph, graph


def generate_lemos(vertex_min, vertex_max, random_edges_factor=2):
    # randomly choose number of nodes in the graph
    no_of_nodes = random.randint(vertex_min,vertex_max)
    no_of_colors = 3

    # generate empty graph
    nodes = list(range(no_of_nodes))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    # variables and basic clauses for faster sat solving
    variables = compute_variables(no_of_nodes, no_of_colors)
    basic_clauses = compute_basic_clauses(no_of_nodes,no_of_colors, variables=variables)
    edge_constraints = []

    # start with a number of random edges
    sat = False
    no_of_random_edges = int(random_edges_factor * no_of_nodes)
    while not sat:
        graph.add_nodes_from(nodes)
        # start with some random edges
        nodes_from = random.choices(nodes, k=no_of_random_edges)
        nodes_to = random.choices(nodes, k=no_of_random_edges)
        pairs = zip(nodes_from, nodes_to)
        graph.add_edges_from(pairs)  # duplicate edges are ignored
        # remove any selfloop edges that might have been inserted
        graph.remove_edges_from(nx.selfloop_edges(graph))

        edge_constraints = compute_edge_clauses(
            graph, no_of_colors, variables=variables)
        sat, _ = find_coloring(graph, no_of_colors, fixed_clauses=basic_clauses,
                                      variables=variables, edge_constraints=edge_constraints)
        # if UNSAT, start over with less edges 
        if not sat:
            # should happen rarely as long as random_edges_factor < 2.3
            no_of_random_edges = int((no_of_random_edges/4)*3)
            graph = nx.Graph()
            print("oh no!")

    # add random edges until there is no longer a 3-coloring
    last_edge = None
    while sat:
        # add a random edge
        edge = random.sample(nodes,2)
        if not graph.has_edge(edge[0],edge[1]):
            last_edge = edge
            graph.add_edge(edge[0],edge[1])
            edge_constraints += compute_single_edge_constraint(edge, no_of_colors, variables)

            # check if still SAT
            sat, _ = find_coloring(graph, no_of_colors,fixed_clauses=basic_clauses, 
                            variables=variables, edge_constraints=edge_constraints)

    neg_graph = graph
    pos_graph = graph.copy()
    pos_graph.remove_edge(last_edge[0],last_edge[1])

    return pos_graph, neg_graph


def main():
    args = parser.parse_args()

    path = os.path.join('data', args.name)
    if not os.path.exists(path):
        os.mkdir(path)

    pos_path = os.path.join(path, 'positive')
    neg_path = os.path.join(path, 'negative')
    if not os.path.exists(pos_path):
        os.mkdir(pos_path)
    if not os.path.exists(neg_path):
        os.mkdir(neg_path)


    block_size = args.block_size

    # generate graphs
    blocks = max(1,int(args.graph_count / (block_size * 2))) # only generate full blocks, but at least one (for testing mainly)
    for block in range(blocks):
        print(f'Generating {min(block_size,args.graph_count)} pairs of graphs. Block {block} with {block_size} pairs of graphs') 

        positive = []
        negative = []
        sizes = []
        for i in tqdm(range(min(block_size,args.graph_count))):
            if not args.lemos:
                pos_graph, neg_graph = generate_lemos(args.vertex_count,args.vertex_count)
                # pos_graph, neg_graph = generate(args.vertex_count, random_edges_factor=args.random_edges_factor)
            else:
                pos_graph, neg_graph = generate_lemos(40,60)
            positive.append(pos_graph)
            negative.append(neg_graph)
            sizes.append(pos_graph.size())

        pos_edges = sum(sizes)/len(sizes)
        neg_edges = pos_edges + 1
            
        data_utils.add_block(positive, pos_path, pos_edges)
        data_utils.add_block(negative, neg_path, neg_edges)
        print(f"The graphs generated have an average number of edges of {pos_edges}")
        
        print(f" {positive_counter} positive instances took {time_for_positive_instances:.2f} seconds")
        print(f" {negative_counter} negative instances took {time_for_negative_instances:.2f} seconds")

if __name__ == '__main__':
    main()
