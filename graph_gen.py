import numpy as np
import pandas as pd
from dataprep import data_r
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False
})

def buildKnnGraph(data, k):

    X = data.iloc[:, 1:-1].values      # features
    ids = data.iloc[:, 0].values       # first column (galaxy IDs)
    y = data.iloc[:, -1].values        # labels (last column)

    N = X.shape[0]                  # k = sqrt(num samples)
    k = k
    
    # Build KNN
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Build edge list
    src = np.repeat(np.arange(N), k)                
    dst = indices[:, 1:].reshape(-1)                 

    # Make bidirectional edges:
    edge_index = np.vstack([src, dst])
    edge_index = np.hstack([edge_index, edge_index[::-1]])

    return ids, X, y, edge_index

def plotGraphByLabel(ids, X, y, edge_index, node_size=40):
    
    labels_map = {
    0: "UDGs",
    1: "LSBs",
    2: "HSBs",
    3: "Dwarfs"
    }
    palette = ["#007ea7", "#80ced7", "#9ad1d4", "#69d1c5"]
    
    edges = edge_index.T

    N = X.shape[0]

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Attach node attributes
    for i in range(N):
        G.nodes[i]["x"] = X[i]
        G.nodes[i]["y"] = y[i]
        G.nodes[i]["id"] = ids[i]

    # Add edges
    G.add_edges_from(edges)

    # Plot
    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)
    node_colors = [palette[int(c)] for c in y]
    nx.draw(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        with_labels=False,
        edge_color="#d0d0d0",
        width=0.5
    )


    legend_handles = [
    mpatches.Patch(color=palette[i], label=labels_map[i])
    for i in range(4)
    ]
    plt.title("Graph structure created from the dataset (K=11)")
    plt.legend(
    handles=legend_handles,
    title="Galaxy Class",
    loc="lower right",
    frameon=True,
    fontsize=10
    )
    plt.savefig(f"datagraph.png", dpi=400, bbox_inches="tight")
    plt.close()


ids, X, y, edges = buildKnnGraph(data_r, 11)

plotGraphByLabel(ids,X,y,edges)

def plotSubgraph(data, num_nodes=150, node_size=40):
    
    N = data.num_nodes
    num_nodes = min(num_nodes, N)

    # random sample of node indices
    subset = torch.randperm(N)[:num_nodes].tolist()

    
    G_full = to_networkx(data, to_undirected=True)

    
    G_sub = G_full.subgraph(subset).copy()

    
    subset_labels = data.y[subset].cpu().numpy()

   
    pos = nx.spring_layout(G_sub, seed=42)

    plt.figure(figsize=(7, 7))

    nx.draw(
        G_sub,
        pos,
        node_color=subset_labels,
        cmap="tab10",
        node_size=node_size,
        edge_color="gray",
        width=0.6,
        with_labels=False
    )

    plt.show()

