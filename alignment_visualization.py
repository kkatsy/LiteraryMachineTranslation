import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

source_sentence = 'Hell has no fury like a woman scorned'
gt_sentence = 'У ада нет ярости как у отвергнутой женщины'


source_len = len(source_sentence.split())
gt_len = len(gt_sentence.split())
source_nodes = ['A'+ str(s1) for s1 in range(source_len)]
gt_nodes = ['B'+ str(s2) for s2 in range(gt_len)]

G = nx.Graph()
alignment = [(0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (7, 6)]
edges = []
labels_dict = {}
nodes = []
for s1, s2 in alignment:
    n1 = 'A'+ str(s1)
    n2 = 'B'+ str(s2)

    if n1 not in labels_dict:
        nodes.append(n1)
        labels_dict[n1] = source_sentence.split()[s1]

    if n2 not in labels_dict:
        nodes.append(n2)
        labels_dict[n2] = gt_sentence.split()[s2]

    edges.append((n1, n2))

for n in source_nodes:
    if n not in nodes:
        labels_dict[n] = source_sentence.split()[int(n[1])]

for n in gt_nodes:
    if n not in nodes:
        labels_dict[n] = gt_sentence.split()[int(n[1])]

G.add_nodes_from(source_nodes, bipartite=0) # Add the node attribute "bipartite"
G.add_nodes_from(gt_nodes, bipartite=1)
G.add_edges_from(edges)

pos = dict()
pos.update((n, (i, 2)) for i, n in enumerate(source_nodes))
pos.update((n, (i, 1)) for i, n in enumerate(gt_nodes))

nx.draw_networkx(G, pos=pos, with_labels=True, node_color='white', font_weight='semibold', width=2,
                 edge_color='forestgreen', labels=labels_dict)

plt.show()
