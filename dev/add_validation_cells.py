import json
from pathlib import Path

# Read the notebook
nb_path = Path(__file__).parent / "benchmark_loading.ipynb"
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Read the new cell content
validation_file = Path(__file__).parent / "validation_cells_content.py"
with open(validation_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split into markdown and code cells
lines = content.strip().split('\n')
cells = []
current_cell = []
current_type = None

for line in lines:
    if line.startswith('# %% [markdown]'):
        if current_cell and current_type:
            cells.append((current_type, current_cell))
        current_cell = []
        current_type = 'markdown'
    elif line.startswith('# %%'):
        if current_cell and current_type:
            cells.append((current_type, current_cell))
        current_cell = []
        current_type = 'code'
    else:
        if current_type == 'markdown' and line.startswith('# '):
            current_cell.append(line[2:])  # Remove '# ' prefix
        else:
            current_cell.append(line)

# Add the last cell
if current_cell and current_type:
    cells.append((current_type, current_cell))

# Convert to notebook cells
new_cells = []
for cell_type, lines in cells:
    source = '\n'.join(lines).strip() + '\n'

    if cell_type == 'markdown':
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [source]
        })
    else:
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [source]
        })

# Add to notebook
nb['cells'].extend(new_cells)

# Write back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added {len(new_cells)} validation cells to benchmark_loading.ipynb")
print("Section 13.11: Visual Quality Check of Scan-Phase Correction")
