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
                           capacity=int(capacity)/1000)
    return graph


def parse_demands(file_path, num_nodes):
    demands = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            demand_values = line.strip().split()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    demand = float(demand_values[i * num_nodes + j])
                    if i != j:
                        demands.append((i + 1, j + 1, demand))
    return demands


def draw(graph):
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=800, node_color='lightblue',
            edge_color='gray', font_size=12, font_weight='bold')

    labels = nx.get_edge_attributes(graph, 'capacity')
    for key in labels:
        labels[key] = f'{labels[key]} Tbps'
    nx.draw_networkx_edge_labels(graph, pos, font_size=8, edge_labels=labels)
    plt.show()


def create_mcf_model(graph, demands):
    return


def draw_flow_allocations(graph, demands, flow_vars):
    edge_flows = {}
    for u, v, c in graph.edges(data='capacity'):
        edge_flow = sum(flow_vars[u, v, i, j].x for i, j, d in demands)
        edge_flows[u, v] = round(edge_flow, 2)

    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=800, node_color='lightblue',
            edge_color='gray', font_size=12, font_weight='bold')

    labels = nx.get_edge_attributes(graph, 'capacity')
    for key in labels:
        labels[key] = f'{labels[key]} Gbps\n({edge_flows[key]} Gbps)'
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def main():
    topology_file = 'topology.txt'
    demands_file = 'demand.txt'

    graph = parse_topology(topology_file)
    num_nodes = graph.number_of_nodes()
    demands = parse_demands(demands_file, num_nodes)

    mcf_model, flow_vars = create_mcf_model(graph, demands)
    if mcf_model.status == GRB.OPTIMAL:
        print("Total throughput:", mcf_model.objVal, "Gbps")
        draw_flow_allocations(graph, demands, flow_vars)
    else:
        print("No optimal solution found.")


if __name__ == "__main__":
    main()
