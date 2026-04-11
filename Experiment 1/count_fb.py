import csv

edges = list(csv.reader(open('MUSAE-master/input/edges/facebook_edges.csv')))
edges = edges[1:]
nodes = set()
for e in edges:
    nodes.add(e[0])
    nodes.add(e[1])
print('Nodes:', len(nodes))
print('Edges:', len(edges))
