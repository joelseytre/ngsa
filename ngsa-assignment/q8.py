import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n = 1000
p = 0.009
G = nx.fast_gnp_random_graph(n, p)
degrees = [d for n, d in nx.degree(G)]

print("Mean degree: %.2f" % np.mean(degrees))

values = sorted(np.unique(degrees))
counts = [degrees.count(val) for val in values]

please_plot = True
if please_plot:
    plt.plot(values, counts)
    ax = plt.gca()
    plt.title("Degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()