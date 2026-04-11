import json

try:
    with open('facebook_analysis.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            # check if this is the pip install cell
            has_networkx_install = any('pip install networkx' in line for line in source)
            has_scipy = any('pip install scipy' in line for line in source)
            if has_networkx_install and not has_scipy:
                # Insert scipy install right after networkx install
                for i, line in enumerate(source):
                    if 'pip install networkx' in line:
                        source.insert(i + 1, '!pip install scipy\n')
                        break
                cell['source'] = source
                break
                
    with open('facebook_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Successfully patched notebook.")
except Exception as e:
    print(f"Error patching notebook: {e}")
