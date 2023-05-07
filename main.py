import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from itertools import combinations, islice
# import pygraphviz as pgv
import random
import sys
# B4 Topology

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


def parse_demands(file_path, num_nodes):
    demands = {}
    print(list(demands.keys()))
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            demand_values = line.strip().split()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    demand = float(demand_values[i * num_nodes + j])
                    if i != j:
                        # The below code needs to be changed to i and j instead of i+1 and j+1 for Sprint topology
                        if (i+1, j+1) in demands:
                            demands[(i+1, j+1)] = max(demands[(i+1, j+1)], demand)
                        else:
                            demands[(i+1, j+1)] = demand
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
def create_mcf_model(graph, demands, mlu_weight = 0):
    print(enumerate(demands.keys()))
    total_demand = sum(demands.values())
    edge_capacity_dict = {
        (u, v): c for u, v, c in graph.edges(data='capacity')}
    edges, capacity = gp.multidict(edge_capacity_dict)
    
    paths = defaultdict(list)
    for s, d in demands:
        for x in list(islice(nx.shortest_simple_paths(graph, s, d), 10)):
            if s != d:
                paths[s, d].append(list(combinations(x, 2)))
    flow = {}
    model = gp.Model("mcf")

    # Maps edges to flow variables
    edges_to_flow = defaultdict(list)

    # List of (source, destination, and id) for each path between every source and destination
    iter_sd = []

    # Create flow variables for each edge within each path between source and destination
    for s, d in paths:
        for idx, path in enumerate(paths[s, d]):
            iter_sd.append((s, d, idx))
            for i, j in path:
                flow[s, d, i, j, idx] = model.addVar(
                    vtype=GRB.CONTINUOUS, name=f'flow_{s}_{d}_{i}_{j}_{idx}')
                edges_to_flow[i, j].append((s, d, idx))

    # Conservation Constraints
    for s, d in paths:
        for idx, path in enumerate(paths[s, d]):
            # Can be made more efficient by only comparing with one right before it
            for i in range(1, len(path)):
                i1, j1 = path[i - 1][0], path[i - 1][1]
                i2, j2 = path[i][0], path[i][1]
                model.addConstr(flow[s, d, i1, j1, idx] == flow[s, d, i2, j2, idx], f'flow_{s}_{d}_{i1}_{j1}_{i2}_{j2}_{idx}')

    # Capacity constraints
    model.addConstrs((((gp.quicksum(
        flow[s, d, i, j, idx] for s, d, idx in edges_to_flow[i, j])) + (gp.quicksum(
        flow[s, d, j, i, idx] for s, d, idx in edges_to_flow[j, i]))) <= capacity[i, j] for i, j in edges), "CAPACITY")

    # MLU Capacity constraints
    mlu = model.addVar(vtype=GRB.CONTINUOUS, name="MLU")
    model.addConstrs((((gp.quicksum(
        flow[s, d, i, j, idx] for s, d, idx in edges_to_flow[i, j])) + (gp.quicksum(
        flow[s, d, j, i, idx] for s, d, idx in edges_to_flow[j, i])))/capacity[i, j] <= mlu for i, j in edges), "MLU CAP")

    # Demand constraints (using first edge as representing flow of path)
    for s, d in paths:
        model.addConstr(gp.quicksum(flow[s, d, paths[s, d][idx][0][0], paths[s, d][idx][0][1], idx] for idx in range(len(paths[s, d]))) <= demands[s, d], f"DEMAND_{s}_{d}")

    # Non-zero demand constraints
    for s, d in paths:
        model.addConstr(gp.quicksum(flow[s, d, paths[s, d][idx][0][0], paths[s, d][idx][0][1], idx] for idx in range(len(paths[s, d]))) >= 0, f"MIN_FLOW_{s}_{d}")

    model.setObjective((1 - mlu_weight) * gp.quicksum(flow[s, d, paths[s, d][idx][0][0], paths[s, d][idx][0][1], idx] for s, d, idx in iter_sd) - mlu_weight * mlu * total_demand, GRB.MAXIMIZE)

    model.optimize()

    # Below is code used for analysis
    demand_met = defaultdict(int) # Maps source and destination to demand met
    edge_flow = defaultdict(int) # Maps edges to flow along those edges (bidirectional edges)
    total_demand_met = 0
    seen = set() # To check if path has already been accounted for

    # for v in model.getVars():
    #     if v.X != 0:
    #         if v.VarName != 'MLU':
    #             o, s, d, a, b, i = (v.VarName).split('_')
    #             if (s, d, i) not in seen:
    #                 seen.add((s, d, i))
    #                 demand_met[s, d] += v.X
    #                 total_demand_met += v.X
    #             if int(a) < int(b):
    #                 edge_flow[a, b] += v.X
    #             else:
    #                 edge_flow[b, a] += v.X
    #             # print('%s %g' % (v.VarName, v.X))
    #         else:
    #             print('%s %g' % (v.VarName, v.X))
    #             print('%s %g' % (v.VarName, v.X * 1000000000))
            
    # # Print flow through graph 
    # for x in demand_met:
    #     print (x,':',demand_met[x])

    # # Code to print graph starts
    # out_graph = nx.Graph()
    # seen_edges = set()
    # max_capacity = 0
    # for x in sorted(edge_flow):
    #     out_graph.add_edge(int(x[0]), int(x[1]), capacity=int(edge_flow[x]))
    #     seen_edges.add((int(x[0]), int(x[1])))
    #     max_capacity = max(max_capacity, edge_flow[x])
    
    # for edge in edges:
    #     if edge[0] < edge[1] and (edge[0], edge[1]) not in seen_edges:
    #         out_graph.add_edge(int(edge[0]), int(edge[1]), capacity=0)

    # out_edges = out_graph.edges()
    # pos = nx.circular_layout(out_graph)
    # weights = [out_graph[u][v]['capacity'] / max_capacity * 10 + 1 for u,v in out_edges]
    # nx.draw(out_graph, pos, with_labels=True, node_size=800, node_color='lightblue',
    #         edge_color='gray', font_size=12, font_weight='bold', width=weights)

    # labels = nx.get_edge_attributes(out_graph, 'capacity')
    # for key in labels:
    #     labels[key] = f'{labels[key]} Gbps'
    # nx.draw_networkx_edge_labels(out_graph, pos, font_size=8, edge_labels=labels)
    # plt.show()
    # # Code to print graph ends

    print("Demand met:", total_demand_met, "out of", total_demand)
    print("Percent of demand met:", total_demand_met / total_demand)
    return model, flow

def draw_flow_allocations(graph, flow):
    # Copy graph
    graph_copy = nx.DiGraph(graph)

    # Flow attributes
    for s, d, a, b, i in flow:
        if flow[s, d, a, b, i].x > 0:
            if graph_copy.has_edge(a, b):
                if 'flow' in graph_copy[a][b]:
                    graph_copy[a][b]['flow'] += flow[s, d, a, b, i].x
                else:
                    graph_copy[a][b]['flow'] = flow[s, d, a, b, i].x
            else:
                graph_copy.add_edge(a, b, flow=flow[s, d, a, b, i].x)

    # Convert graph (networkx --> pygraphviz)
    A = nx.nx_agraph.to_agraph(graph_copy)

    # Graph appearance
    A.node_attr.update(color="lightblue", style="filled", fontsize=14)
    A.edge_attr.update(fontsize=10)

    # Flow labels
    for a, b in A.edges():
        flow_value = graph_copy[int(a)][int(b)]['flow']
        A.get_edge(a, b).attr['label'] = f"{flow_value:.2f}"

    # Render + Display
    A.draw("flow_allocations.png", prog='dot')

def main(args):
    if len(args) > 1 and args[1] == 'extra':
        times = []
        for i in range(1, 9):
            size = i * 50
            graph = generate_graph(size)
            demands = generate_demands(size)
            model, flow = create_mcf_model(graph, demands)
            times.append(model.Runtime)
            print(size)
            print(model.Runtime)
        print(times)
        return times
    else:
        topology_file = 'topology.txt'
        demands_file = 'demand.txt'

        graph = parse_topology(topology_file)
        num_nodes = graph.number_of_nodes()
        demands = parse_demands(demands_file, num_nodes)

        mlu_weight = 0
        mcf_model, flow = create_mcf_model(graph, demands, mlu_weight)
        # draw_flow_allocations(graph, flow)


if __name__ == "__main__":
    main(sys.argv)

