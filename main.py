import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import pygraphviz as pgv

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
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            demand_values = line.strip().split()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    demand = float(demand_values[i * num_nodes + j])
                    if i != j:
                        demands[(i+1, j+1)] = demand
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


def create_mcf_model(graph, demands):
    edge_capacity_dict = {
        (u, v): c for u, v, c in graph.edges(data='capacity')}
    edges, capacity = gp.multidict(edge_capacity_dict)
    paths = {(s, d): list(nx.all_simple_edge_paths(graph, s, d))
             for s, d in demands}

    flow = {}
    model = gp.Model("mcf")

    edges_to_flow = defaultdict(list)
    sd_to_flow = defaultdict(list)
    rep_sd = {}
    rep_vals = defaultdict(list)
    enum_sd = []
    iter_sd = []

    # create flow variables + update dicts/lists
    for s, d in paths:
        for i, path in enumerate(paths[s, d]):
            enum_sd.append((s, d, i))
            for a, b in path:
                if (s, d, i) not in rep_sd:
                    rep_sd[s, d, i] = a, b
                    iter_sd.append((s, d, a, b, i))
                flow[s, d, a, b, i] = model.addVar(
                    vtype=GRB.CONTINUOUS, name=f'flow_{s}_{d}_{a}_{b}_{i}')
                sd_to_flow[s, d].append((a, b, i))
                edges_to_flow[a, b].append((s, d, i))
                rep_vals[s, d, i].append((a, b))

    # Flow Constraints
    for s, d, i in enum_sd:
        for a1, b1 in rep_vals[s, d, i]:
            for a2, b2 in rep_vals[s, d, i]:
                model.addConstr(flow[s, d, a1, b1, i] == flow[s, d,
                                a2, b2, i], f'flow_{s}_{d}_{a1}_{b1}_{a2}_{b2}_{i}')

    # Capacity constraints
    model.addConstrs(((gp.quicksum(
        flow[s, d, a, b, i] for s, d, i in edges_to_flow[a, b])) <= capacity[a, b] for a, b in edges), "CAPACITY")

    # MLU Capacity constraints
    mlu = model.addVar(vtype=GRB.CONTINUOUS, name="MLU")
    model.addConstrs(((gp.quicksum(
        flow[s, d, a, b, i] for s, d, i in edges_to_flow[a, b])) <= mlu for a, b in edges), "MLU CAP")

    # "Demand" constraints
    for s, d in demands:
        model.addConstr(gp.quicksum(flow[s, d, rep_sd[s, d, i][0], rep_sd[s, d, i][1], i] for i in range(
            len(paths[s, d]))) <= demands[s, d], f"DEMAND_{s}_{d}")

    c = 0  # How important minimizing MLU is for optimization
    model.setObjective(gp.quicksum(
        flow[s, d, a, b, i] for s, d, a, b, i in iter_sd) - mlu * c, GRB.MAXIMIZE)

    model.optimize()

    # Testing
    # for v in model.getVars():
    #     if v.X != 0:
    #         print('%s %g' % (v.VarName, v.X))

    return model, flow


def draw_flow_allocations(graph, flow):
    # Adding flow
    for s, d, a, b, i in flow:
        if flow[s, d, a, b, i].x > 0:
            if graph.has_edge(a, b):
                if 'flow' in graph[a][b]:
                    graph[a][b]['flow'] += flow[s, d, a, b, i].x
                else:
                    graph[a][b]['flow'] = flow[s, d, a, b, i].x
            else:
                graph.add_edge(a, b, flow=flow[s, d, a, b, i].x)

    # Convert graph (networkx --> pygraphviz)
    A = nx.nx_agraph.to_agraph(graph)

    # Appearance
    A.node_attr.update(color="lightblue", style="filled", fontsize=14)
    A.edge_attr.update(fontsize=10)

    # Flow labels
    for a, b in A.edges():
        flow_value = graph[int(a)][int(b)]['flow']
        A.get_edge(a, b).attr['label'] = f"{flow_value:.2f}"

    # Render + Display
    A.draw("flow_allocations.png", prog='dot')
    A.view("flow_allocations.png")


def main():
    topology_file = 'topology.txt'
    demands_file = 'demand.txt'

    graph1 = parse_topology(topology_file)
    num_nodes = graph1.number_of_nodes()
    demands = parse_demands(demands_file, num_nodes)
    # print(demands)
    # draw(graph)
    graph2 = graph1.copy()
    mcf_model, flow = create_mcf_model(graph1, demands)
    print("Total throughput:", mcf_model.objVal, "Gbps")
    draw_flow_allocations(graph2, flow)

    # mcf_model, flow_vars = create_mcf_model(graph, demands)
    # if mcf_model.status == GRB.OPTIMAL:
    #     print("Total throughput:", mcf_model.objVal, "Gbps")
    #     # draw_flow_allocations(graph, demands, flow_vars)
    # else:
    #     print("No optimal solution found.")


if __name__ == "__main__":
    main()
