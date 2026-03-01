import json, sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    ct = c['cell_type']
    print(f"\n{'='*60}")
    print(f"CELL {i} ({ct}) - {len(src)} chars")
    print('='*60)
    print(src[:8000])
    if len(src) > 8000:
        print(f"\n... [TRUNCATED, {len(src)} total chars]")
