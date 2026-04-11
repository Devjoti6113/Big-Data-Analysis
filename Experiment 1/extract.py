import json
try:
    with open('big-data-project-final.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open('parsed_nb.txt', 'w', encoding='utf-8') as out:
        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type', 'unknown')
            out.write(f'## CELL TYPE: {cell_type}\n')
            
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            out.write(source + '\n\n')
    print("Success")
except Exception as e:
    print("Error:", e)
