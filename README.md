# UniqMol
# UniqMol: A program for identifying unique and duplicate/near-duplicate molecular structures

UniqMol is a single‑file command‑line tool that scans a folder of Gaussian `.log` files, groups entries by chemical formula, and filters out duplicate or near‑duplicate conformers while keeping the original basenames. It selects a lowest‑energy representative per duplicate set and can optionally compute symmetry‑aware RMSD and 3D shape similarity for additional context.

## Quick start

```bash
python3 uniqmol.py --logdir . --no-pair-log
```

- `--logdir` specifies the directory that contains the Gaussian `.log` files.
- `--no-pair-log` suppresses the detailed per‑pair report. You can omit this option, but with thousands of comparisons the report can become extremely large.

## Options

- `--logdir PATH`  
  Directory with Gaussian `.log` files.

- `--filelist FILE`  
  Plain‑text file listing basenames (without the `.log` extension), one per line. Only those logs will be compared.

- `--no-pair-log`  
  Skip writing the large per‑pair comparison files (faster and saves disk space for big runs).

- `--symm-rmsd` / `--symm-reflection`  
  Enable informational, symmetry‑aware RMSD via RDKit. `--symm-reflection` also tests reflected coordinates (requires `--symm-rmsd`).

- `--no-shape`  
  Skip RDKit 3D shape Tanimoto calculation (informational only).

Notes:
- RDKit‑dependent features are optional; if RDKit is not installed, the script will skip them and proceed.
- Shape and symmetry metrics are informational and not part of the duplicate decision.

## What gets written

- `unique_isomers_<FORMULA>.txt` — details for the chosen unique representatives in each formula group.
- `unique_isomers_list_update.txt` — flat list of all unique basenames across groups.
- `comparisons_<FORMULA>.txt` — per‑pair audit log (omitted if you use `--no-pair-log`).
- `original_name_map.txt` — formula → original basename → full path mapping.
- `problem_to_parse.txt` — files with parsing/geometry/RMSD issues.

## How duplicates are detected (summary)

1. Group inputs by chemical formula:
   - Prefer the Gaussian “Stoichiometry” line.
   - Fall back to a Hill‑system formula derived from parsed geometry.
2. Within each group, compare pairs using property thresholds (defaults in the script):
   - Energy (0.1 kcal/mol), dipole (0.1 D), max vibrational frequency difference (1 cm⁻¹),
     HOMO/LUMO (0.1 eV), rotational constants (0.5 GHz), and inertia moment differences.
3. If electronic properties agree within thresholds, check RMSD (Open Babel):
   - High RMSD with otherwise matching properties is treated as a rotamer duplicate; the lowest‑energy conformer is kept.

You can adjust the thresholds by editing the `THRESHOLDS` dictionary near the top of `uniqmol.py`.


## Examples

- Scan all logs in a folder:
  ```bash
  python uniqmol.py --logdir ./logs --no-pair-log
  ```

- Use an explicit file list (basenames only, one per line):
  ```bash
  python uniqmol.py --logdir ./logs --filelist filelist.txt
  ```

## Tips

- With large datasets, prefer `--no-pair-log` to avoid huge audit files.
- If you care about the audit trail, leave `--no-pair-log` off, but be mindful of disk usage.
- If RDKit isn’t installed, the script will print a note and skip symmetry/shape features.

---

Developed by Emmanouil Semidalas in collaboration with Amir Karton.

# Citation
Emmanouil Semidalas and Amir Karton, Benchmarking Isomerization Energies for C5--C7 Hydrocarbons: The ISOC7 Database, 2025


# License
GNU General Public License v3.0
