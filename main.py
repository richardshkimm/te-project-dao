import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

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

    T = {(s, d): list(nx.all_simple_paths(graph, s, d)) for s, d in demands}

    model = gp.Model("MCF")

    # Variables
    flow = model.addVars(edges, name="FLOW")
    # flow = {}
    # for i, j, d in demands:
    #     for u, v, c in graph.edges(data='capacity'):
    #         flow[u, v, i, j] = model.addVar(
    #             vtype=GRB.continuous, name=f'flow_{u}_{v}_{i}_{j}')

    # Constraints
    model.addConstr(
        (flow.sum(i, j) <= capacity[i, j] for i, j in edges), "CAPACITY")

    model.addConstr((flow.sum(T[d]) <= demands[d]
                    for d in demands), "DEMANDS")

    # Objective
    throughput = flow.sum(d for d in demands)
    model.setObjective(throughput, GRB.MAXIMIZE)

    # Optimize
    model.optimize()

    return model, flow


def draw_flow_allocations(graph, demands, flow_vars):
    return


def main():
    topology_file = 'topology.txt'
    demands_file = 'demand.txt'

    graph = parse_topology(topology_file)
    num_nodes = graph.number_of_nodes()
    demands = parse_demands(demands_file, num_nodes)
    # print(demands)
    # draw(graph)
    create_mcf_model(graph, demands)

    # mcf_model, flow_vars = create_mcf_model(graph, demands)
    # if mcf_model.status == GRB.OPTIMAL:
    #     print("Total throughput:", mcf_model.objVal, "Gbps")
    #     # draw_flow_allocations(graph, demands, flow_vars)
    # else:
    #     print("No optimal solution found.")


if __name__ == "__main__":
    main()
