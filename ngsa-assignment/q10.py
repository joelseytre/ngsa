import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

G = nx.read_edgelist("CA-GrQc.txt")
G = max(nx.connected_component_subgraphs(G), key=len)

nodes = [n for n, d in nx.degree(G)]
degrees = [d for n, d in nx.degree(G)]

# sorts the nodes based on degrees
nodes = [n for _, n in sorted(zip(degrees, nodes), reverse=True)]
n = len(nodes)

# keep 20%
large_nodes = nodes[:int(0.2*n)]

G1 = G.copy()
G2 = G.copy()

starting_length = len(nodes)
current_length = starting_length
step = int(0.02 * starting_length)
GCCs_failure = []
rests_failure = []
totals = []
while current_length >= 0.8 * starting_length:
    gcc = max(nx.connected_component_subgraphs(G1), key=len)
    rest = sorted(nx.connected_component_subgraphs(G1), key=len, reverse=True)
    rest.remove(rest[0])
    GCCs_failure += [len(gcc.nodes)]
    rests_failure += [np.mean([len(rest[i].nodes) for i in range(len(rest))])]
    totals += [len(nodes)]

    to_remove = random.sample(nodes, k=step)
    for node in to_remove:
        G1.remove_node(node)
    nodes = [n for n, d in nx.degree(G1)]
    current_length = len(nodes)

GCCs_attack = []
rests_attack = []
steps = []
current_steps = 0
current_length = starting_length
nodes = [n for n, d in nx.degree(G2)]
while len(large_nodes) > 0:
    gcc = max(nx.connected_component_subgraphs(G2), key=len)
    rest = sorted(nx.connected_component_subgraphs(G2), key=len, reverse=True)
    rest.remove(rest[0])
    GCCs_attack += [len(gcc.nodes)]
    print(len(rest))
    rests_attack += [np.mean([len(rest[i].nodes) for i in range(len(rest))])]
    totals += [len(nodes)]
    steps += [current_steps]
    current_steps += 0.02

    to_remove = random.sample(large_nodes, k=min(step, len(large_nodes)))
    for node in to_remove:
        G2.remove_node(node)
        large_nodes.remove(node)
    nodes = [n for n, d in nx.degree(G2)]
    current_length = len(nodes)

GCCs_failure = np.array(GCCs_failure) / GCCs_failure[0]
rests_failure = np.array(rests_failure) / rests_failure[1]
GCCs_attack = np.array(GCCs_attack) / GCCs_attack[0]
rests_attack = np.array(rests_attack) / rests_attack[1]
rests_attack[0] = 1
rests_failure[0] = 1

print(rests_attack)
print(rests_failure)

ax1 = plt.gca()
plt.ylabel("Nodes (GCC)")
plt.xlabel("Percentage of nodes missing")
ax2 = ax1.twinx()
plt.ylabel("Nodes (rest)")
ax1.set_ylim([0, 2*GCCs_attack[0]])
ax2.set_ylim([-2, 2])
g_f, = ax1.plot(steps, GCCs_failure, '-xg', label='GCC failure')
r_f, = ax2.plot(steps, rests_failure, '-og', label='rest failure')
g_a, = ax1.plot(steps, GCCs_attack, '-xb', label='GCC attack')
r_a, = ax2.plot(steps, rests_attack, '-ob', label='rest attack')
plt.legend(handles=[g_f, r_f, g_a, r_a])
plt.show()