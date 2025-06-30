# Manual for run_cluster.py

## What it does
* This script calculates the ORR (Oxygen Reduction Reaction) overpotential for a platinum cluster.

## How to run
```bash
python run_cluster.py
```

## What happens
1. Creates a Pt octahedron cluster (3 layers)
2. Tests different adsorbate positions for HO2, O, and OH
3. Calculates energy changes for each reaction step
4. Outputs the overpotential and limiting potential

## Output
* The script will print:
  - ORR overpotential (V)
  - Limiting potential (V)  
  - Reaction free energy changes

* Results are saved to the `result/Pt/` directory.

## Calculation conditions

### Basic settings
```python
outdir = str(Path(__file__).parent / "result" / "Pt")    # Output directory
overwrite = True                                         # Overwrite existing results
log_level = "INFO"                                      # Log level (DEBUG/INFO/WARNING/ERROR)
calculator = "mace"                                     # Calculator type
yaml_path = str(Path(__file__).parent / "vasp.yaml")   # VASP settings file
```

### How to modify the cluster
```python
cluster = Octahedron(symbol="Pt", length=3, cutoff=0)
```
- `symbol`: Metal type ("Pt", "Au", "Pd", etc.)
- `length`: Cluster size (number of layers)
- `cutoff`: Remove atoms within cutoff distance

### How to modify the adsorbate positions
```python
orr_adsorbates: Dict[str, List[Tuple]] = {
    "HO2": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
    "O":   [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
    "OH":  [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
}
```
* Numbers represent atom indices on the cluster surface where adsorbates will be placed.

### VASP settings
* If you are using and want to modify the VASP condition, edit `vasp.yaml` to modify DFT calculation parameters like energy cutoff, k-points, etc.
  