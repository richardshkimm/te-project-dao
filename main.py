import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from itertools import combinations, islice
import random


def parse_topology(file_path):
    graph = nx.DiGraph()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('to_node'):
                continue
            to_node, from_node, capacity, prob_failure = line.strip().split()
            graph.add_edge(int(to_node), int(from_node),
                           capacity=int(capacity))
    return graph


def parse_demands(file_path, num_nodes, is_sprint = False):
    demands = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            demand_values = line.strip().split()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    demand = float(demand_values[i * num_nodes + j])
                    if i != j:
                        k = 0
                        if not is_sprint: # adjusting indexing for other topologies
                            k = 1  
                        if (i+k, j+k) in demands:
                            demands[(i+k, j+k)] = max(demands[(i+k, j+k)], demand)
                        else:
                            demands[(i+k, j+k)] = demand
    return demands


def generate_graph(size):
    graph = nx.erdos_renyi_graph(size,0.5,directed=False)
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(size,0.5,directed=False)
    nx.set_edge_attributes(graph, values = 5000000, name = 'capacity')
    return graph


def generate_demands(size):
    demands = {}
    for i in range(size):
        for j in range(size):
            demands[(i,j)] = random.uniform(1000,50000)
    return demands


def draw(graph):
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=800, node_color='lightblue',
            edge_color='gray', font_size=12, font_weight='bold')

    labels = nx.get_edge_attributes(graph, 'capacity')
    for key in labels:
        labels[key] = f'{labels[key]} Gbps'

    nx.draw_networkx_edge_labels(graph, pos, font_size=8, edge_labels=labels)
    plt.show()

# mlu_weight is how important minimizing MLU is for optimization
def create_mcf_model(graph, demands, mlu_weight = 0, extra_credit = False):
    total_demand = sum(demands.values())
    edge_capacity_dict = {
        (u, v): c for u, v, c in graph.edges(data='capacity')}
    edges, capacity = gp.multidict(edge_capacity_dict)
    
    paths = defaultdict(list)
    for s, d in demands:
        if not extra_credit:
            paths = {(s, d): list(nx.all_simple_edge_paths(graph, s, d)) for s, d in demands}
        else:
            for x in list(islice(nx.shortest_simple_paths(graph, s, d), 10)):
                if s != d:
                    paths[s, d].append(list(combinations(x, 2)))

    flow = {}
    model = gp.Model("mcf")

    edges_to_flow = defaultdict(list) # Maps edges to flow paths
    
    for s, d in paths:
        for idx, path in enumerate(paths[s, d]):
            for i, j in path:
                edges_to_flow[i, j].append((s, d, idx))
            flow[s, d, idx] = model.addVar(
                vtype=GRB.CONTINUOUS, name=f'flow_{s}_{d}_{idx}')
    
    # List of (source, destination, and id) for each path between every source and destination
    model.addConstrs(((gp.quicksum(flow[s, d, idx] for s, d, idx in edges_to_flow[i, j])) + (gp.quicksum(
        flow[s, d, idx] for s, d, idx in edges_to_flow[j, i])) <= capacity[i, j] for i, j in edges), "CAPACITY")

    # MLU Capacity constraints
    mlu = model.addVar(vtype=GRB.CONTINUOUS, name="MLU")
    model.addConstrs(((((gp.quicksum(flow[s, d, idx] for s, d, idx in edges_to_flow[i, j])) + (gp.quicksum(
        flow[s, d, idx] for s, d, idx in edges_to_flow[j, i])))/capacity[i, j]) <= mlu for i, j in edges), "MLU CAP")

    for s, d in paths:
        # Demand constraints
        model.addConstr(gp.quicksum(flow[s, d, idx] for idx in range(len(paths[s, d]))) <= demands[s, d], f"DEMAND_{s}_{d}")
        # Non-negative demand constraints
        model.addConstr(gp.quicksum(flow[s, d, idx] for idx in range(len(paths[s, d]))) >= 0, f"NONNEGATIVE_{s}_{d}")

    # Set objective
    if mlu_weight == 0:
        model.setObjective(gp.quicksum(gp.quicksum(flow[s, d, idx] for idx in range(len(paths[s,d]))) for s, d in paths), GRB.MAXIMIZE)
    else:
        model.setObjective((1 - mlu_weight) * gp.quicksum(gp.quicksum(flow[s, d, idx] for idx in range(len(paths[s,d]))) for s, d in paths) - mlu_weight * mlu * total_demand, GRB.MAXIMIZE)

    model.optimize()

    # Below is code used for analysis
    edge_flow = defaultdict(int) # Maps edges to flow along those edges (bidirectional edges)
    total_demand_met = 0
    
    # Aggregating flow allocations
    for v in model.getVars():
        if v.X != 0:
            if v.VarName != 'MLU':
                _, s, d, idx = (v.VarName).split('_')
                total_demand_met += v.X
                for a, b in paths[int(s), int(d)][int(idx)]:
                    if int(a) < int(b): # Only accounting for one direction
                        edge_flow[a, b] += v.X
                    else:
                        edge_flow[b, a] += v.X
            else:
                final_mlu = v.X

    # # Code to print graph starts
    print_graph = False # set to True to print graph
    if print_graph:
        out_graph = nx.Graph()
        seen_edges = set()
        max_capacity = 0
        for x in sorted(edge_flow):
            out_graph.add_edge(int(x[0]), int(x[1]), capacity=int(edge_flow[x]))
            seen_edges.add((int(x[0]), int(x[1])))
            max_capacity = max(max_capacity, edge_flow[x])
        
        for edge in edges:
            if edge[0] < edge[1] and (edge[0], edge[1]) not in seen_edges:
                out_graph.add_edge(int(edge[0]), int(edge[1]), capacity=0)

        out_edges = out_graph.edges()
        pos = nx.circular_layout(out_graph)
        weights = [out_graph[u][v]['capacity'] / max_capacity * 10 + 1 for u,v in out_edges]
        nx.draw(out_graph, pos, with_labels=True, node_size=800, node_color='lightblue',
                edge_color='gray', font_size=12, font_weight='bold', width=weights)

        labels = nx.get_edge_attributes(out_graph, 'capacity')
        for key in labels:
            labels[key] = f'{labels[key]} Gbps'
        nx.draw_networkx_edge_labels(out_graph, pos, font_size=8, edge_labels=labels)
        plt.show()
    # # Code to print graph ends

    print("\nResults:")
    print("MLU:", final_mlu)
    print("Demand met:", total_demand_met, "out of", total_demand)
    print("Percent of demand met:", total_demand_met / total_demand)

    return model, flow

def main(topology, algorithm):
    topology_file = topology + '/topology.txt'
    demands_file = topology + '/demand.txt'

    graph = parse_topology(topology_file)
    num_nodes = graph.number_of_nodes()

    if topology == 'Sprint':
        demands = parse_demands(demands_file, num_nodes, True)
    else:
        demands = parse_demands(demands_file, num_nodes, False)

    if algorithm == 1:
        mlu_weight = 0
    else:
        mlu_weight = 0.4 # change value for B4 topology to see tradeoff between demand met and MLU

    create_mcf_model(graph, demands, mlu_weight = mlu_weight)

if __name__ == "__main__":
    topology = 'B4' # set to 'Sprint' or 'B4'
    algorithm = 2 # set to 1 (Maximum throughput) or 2 (Minimizing link Utilization)
    main(topology, algorithm)

