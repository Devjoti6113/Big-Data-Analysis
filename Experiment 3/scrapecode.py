import urllib.request
from bs4 import BeautifulSoup
import re

def scrape_snap():
    url = "https://snap.stanford.edu/data/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    print(f"Fetching {url}")
    
    try:
        html = urllib.request.urlopen(req).read()
    except Exception as e:
        with open("result.txt", "w") as f:
            f.write(f"Failed to fetch SNAP: {e}")
        return
        
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    
    datasets = []
    
    for table in tables:
        rows = table.find_all('tr')
        if not rows: continue
        headers = [th.get_text().strip().lower() for th in rows[0].find_all(['th', 'td'])]
        
        node_idx = -1
        edge_idx = -1
        name_idx = 0
        
        for i, h in enumerate(headers):
            if 'nodes' in h: node_idx = i
            if 'edges' in h: edge_idx = i
            
        if node_idx != -1 and edge_idx != -1:
            for row in rows[1:]:
                cols = row.find_all('td')
                if len(cols) > max(node_idx, edge_idx):
                    try:
                        name = cols[name_idx].get_text().strip()
                        nodes_str = re.sub(r'[^\d]', '', cols[node_idx].get_text())
                        edges_str = re.sub(r'[^\d]', '', cols[edge_idx].get_text())
                        if nodes_str and edges_str:
                            nodes = int(nodes_str)
                            edges = int(edges_str)
                            if nodes > 0:
                                avg_degree = 2.0 * edges / nodes
                                datasets.append({
                                    'Name': name,
                                    'Nodes': nodes,
                                    'Edges': edges,
                                    'AvgDegree': avg_degree
                                })
                    except Exception as e:
                        pass
                        
    out = []
    if datasets:
        # Filter for nodes >= 20k
        filtered = [d for d in datasets if d['Nodes'] >= 20000]
        filtered.sort(key=lambda x: x['AvgDegree'])
        out.append("Top Unmodified Datasets with extreme sparsity (Nodes >= 20k):\n")
        out.append(f"{'Dataset Name':<40} {'Nodes':<15} {'Edges':<15} {'Avg Degree'}")
        out.append("-" * 80)
        for d in filtered[:20]:
            out.append(f"{d['Name']:<40} {d['Nodes']:<15} {d['Edges']:<15} {d['AvgDegree']:.2f}")
    else:
        out.append("No datasets matched.")
        
    with open("result.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(out))

if __name__ == "__main__":
    scrape_snap()
