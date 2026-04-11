import json
import random

try:
    with open('facebook_analysis.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if any('plt.figure(figsize=(100, 100))' in line for line in source):
                new_source = [
                    "import random\n",
                    "# Randomly sample 1000 nodes for visualization to prevent graph clutter\n",
                    "sampled_nodes = random.sample(list(G.nodes()), min(1000, G.number_of_nodes()))\n",
                    "G_sub = G.subgraph(sampled_nodes)\n",
                    "\n",
                    "# Initiating Graph Visualization framework\n",
                    "plt.figure(figsize=(20, 20))\n",
                    "\n",
                    "# Applying Spring Layout, limiting iterations to 20 to preserve computation speeds\n",
                    "pos = nx.spring_layout(G_sub, k=0.10, iterations=20)\n",
                    "\n",
                    "# Overlaying the respective properties\n",
                    "nx.draw_networkx_nodes(G_sub, pos, node_size=15, node_color='#1f78b4', alpha=0.8)\n",
                    "nx.draw_networkx_edges(G_sub, pos, width=0.5, alpha=0.1, edge_color='gray')\n",
                    "\n",
                    "plt.title('Facebook Scale Network Subgraph Formations', fontsize=40)\n",
                    "plt.axis('off')\n",
                    "plt.show()\n"
                ]
                cell['source'] = new_source
                break
                
    with open('facebook_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Successfully patched notebook to include random sampling.")
except Exception as e:
    print(f"Error patching notebook: {e}")
