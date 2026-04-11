import pandas as pd
import networkx as nx
df = pd.read_csv('MUSAE-master/input/edges/facebook_edges.csv')
G = nx.from_pandas_edgelist(df, 'id_1', 'id_2')
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
