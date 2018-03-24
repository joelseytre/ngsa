import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
import powerlaw
from datetime import datetime

questions = [2]

G = nx.read_edgelist("CA-GrQc.txt")
connected_components = sorted(nx.connected_component_subgraphs(G),
                              key = len, reverse=True)
gcc = connected_components[0]
if 1 in questions:
    print("Number of nodes: %i" % G.number_of_nodes())
    print("Number of edges: %i" % G.number_of_edges())

    print("\nNumber of CC: %i" % nx.number_connected_components(G))

    cc_size = [g.number_of_nodes() for g in connected_components]
    values = sorted(np.unique(cc_size))
    counts = [cc_size.count(val) for val in values]

    please_plot = True
    if please_plot:
        plt.plot(values, counts)
        ax = plt.gca()
        ax.set_xscale('log')
        plt.title("Distribution of CC size")
        plt.xlabel("CC size")
        plt.ylabel("Count")
        plt.show()

    print("Number of nodes in GCC: %i i.e %.1f%% of total graph"
          % (gcc.number_of_nodes(),
             100*float(gcc.number_of_nodes())/float(G.number_of_nodes())))
    print("Number of edges in GCC: %i i.e %.1f%% of total graph"
          % (gcc.number_of_edges(),
             100*float(gcc.number_of_edges())/float(G.number_of_edges())))
elif 2 in questions:
    nodes = G.nodes
    degrees = [d for n, d in nx.degree(G)]
    print("Min degree: %i" % np.min(degrees))
    print("Max degree: %i" % np.max(degrees))
    print("Mean degree: %.1f" % np.mean(degrees))
    print("Median degree: %i" % np.median(degrees))

    values = sorted(np.unique(degrees))
    counts = [degrees.count(val) for val in values]
    fit = powerlaw.Fit(degrees, discrete=True)
    print('Power law! alpha= ', fit.power_law.alpha, ' - sigma= ', fit.power_law.sigma)
    please_plot = True
    if please_plot:
        plt.plot(values, counts)
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.title("Log-log degree distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.show()
elif 3 in questions:
    to_remove = []
    for edge in gcc.edges:
        if edge[0] == edge[1]:
            to_remove += [edge[0]]
    for n in to_remove:
        gcc.remove_edge(n, n)
    A = np.matrix(nx.to_numpy_matrix(gcc))
    B = A ** 3
    num_triangles = int(np.trace(B)/6)
    print("Total number of triangles: %i" % num_triangles)

    triangle_participation = [B[i, i]/2 for i in range(len(B))]
    values = sorted(np.unique(triangle_participation))
    counts = [triangle_participation.count(val) for val in values]

    please_plot = False
    if please_plot:
        plt.plot(values, counts)
        ax = plt.gca()
        ax.set_xscale('log')
        plt.title("Triangle participation")
        plt.xlabel("Number of triangles (log)")
        plt.ylabel("Count")
        plt.show()

    eigenvalues, eigenvectors = np.array(scipy.sparse.linalg.eigs(A, 1000))
    eigenvalues = eigenvalues.real
    plt.plot(eigenvalues)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.title("Eigenvalue distribution")
    plt.xlabel("Eigenvalue number")
    plt.ylabel("Value")
    plt.show()

    ready_for_long_computation = False
    if ready_for_long_computation:
        num_eig = [1, 10, 50, 100, 500, 1000]
        errors = []
        delays = []
        for k in num_eig:
            start = datetime.now()
            eigenvalues, eigenvectors = np.array(scipy.sparse.linalg.eigs(A, k))
            eigenvalues = eigenvalues.real
            errors += [abs((np.sum(np.power(eigenvalues, 3)) / 6) - num_triangles)
                       / num_triangles]
            time_computation = datetime.now() - start
            delays += [time_computation.minutes
                       + time_computation.seconds
                       + time_computation.microseconds / 1000000]
        plt.subplot(211)
        plt.plot(num_eig, errors, '-bx')
        ax = plt.gca()
        ax.set_xscale('log')
        plt.title("Estimation of triangle participation vs number of eigenvalues")
        plt.xlabel("Number of eigenvalues (log)")
        plt.ylabel("Relative error in number of triangles")

        plt.subplot(212)
        plt.plot(num_eig, delays, '-gx')
        ax = plt.gca()
        ax.set_xscale('log')
        plt.title("Computation time vs number of eigenvalues considered")
        plt.xlabel("Number of eigenvalues (log)")
        plt.ylabel("Computation time")
        plt.show()
