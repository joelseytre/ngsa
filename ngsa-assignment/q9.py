import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


def generate_graph(M):
    g = nx.Graph()
    n = len(M)
    for l in tqdm(range(n)):
        for j in range(n - i):
            if np.random.choice([False, True], p=[1 - M[l, j], M[l, j]]):
                g.add_edge(l, j)
    return g.to_undirected()


A_1 = np.array([[0.99, 0.26], [0.26, 0.53]])

# k = 11 => 4096 nodes
k = 12

start = datetime.now()
A = A_1
for i in range(k):
    A = np.kron(A, A_1)
time_computation = datetime.now() - start
print("Computation time: %s" % time_computation)

start = datetime.now()
k_G = generate_graph(A)
time_computation = datetime.now() - start
print("Reading time: %s" % time_computation)

connected_components_k_G = sorted(nx.connected_component_subgraphs(k_G),
                                  key = len, reverse=True)
gcc_k_G = connected_components_k_G[0]

print("Number of edges in GCC (Kro. model): %i i.e %.1f%% of total graph (%i edges)"
      % (gcc_k_G.number_of_edges(), 100 * float(gcc_k_G.number_of_edges())
         / float(k_G.number_of_edges()), k_G.number_of_edges()))

print("Number of nodes in GCC (Kro. model): %i i.e %.1f%% of total graph (%i nodes)"
      % (gcc_k_G.number_of_nodes(), 100 * float(gcc_k_G.number_of_nodes())
         / float(k_G.number_of_nodes()), k_G.number_of_nodes()))

degrees_k_G = [d for n, d in nx.degree(k_G)]
values_k_G = sorted(np.unique(degrees_k_G))
counts_k_G = [degrees_k_G.count(val) for val in values_k_G]

cc_size_k_G = [g.number_of_nodes() for g in connected_components_k_G]
values_cc_k_G = sorted(np.unique(cc_size_k_G))
counts_cc_k_G = [cc_size_k_G.count(val) for val in values_cc_k_G]

G = nx.read_edgelist("CA-GrQc.txt")
connected_components_G = sorted(nx.connected_component_subgraphs(G),
                                key = len, reverse=True)
gcc_G = connected_components_G[0]

degrees_G = [d for n, d in nx.degree(G)]
values_G = sorted(np.unique(degrees_G))
counts_G = [degrees_G.count(val) for val in values_G]

cc_size_G = [g.number_of_nodes() for g in connected_components_G]
values_cc_G = sorted(np.unique(cc_size_G))
counts_cc_G = [cc_size_G.count(val) for val in values_cc_G]

please_plot = True
if please_plot:
    plt.subplot(211)
    plt.plot(values_G, counts_G)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title("Log-log degree distribution - CA-GrQc")
    plt.xlabel("Degree")
    plt.ylabel("Count")

    plt.subplot(212)
    plt.plot(values_k_G, counts_k_G)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title("Log-log degree distribution - Kronecker model")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

    plt.subplot(211)
    plt.plot(values_cc_G, counts_cc_G)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.title("Distribution of CC size - CA-GrQc")
    plt.xlabel("CC size")
    plt.ylabel("Count")

    plt.subplot(212)
    plt.plot(values_cc_k_G, counts_cc_k_G)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.title("Distribution of CC size - Kronecker model")
    plt.xlabel("CC size")
    plt.ylabel("Count")
    plt.show()