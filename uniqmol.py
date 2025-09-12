#!/usr/bin/env python3
"""
Universal duplicate / near-duplicate conformer detection (keeps original filenames).
- Groups by a chemically derived formula:
      1) Tries Gaussian "Stoichiometry" line.
      2) Falls back to Hill-system formula from parsed geometry.
- Writes unique_isomers_<formula>.txt and unique_isomers_list_update.txt listing original basenames.
- Duplicate filtering logic:
      * Property thresholds (energy 0.1 kcal/mol, dipole 0.1 D, freq 1 cm^-1, HOMO/LUMO 0.1 eV, rot const 0.5 GHz).
      * High RMSD alone (if other properties match) => rotamer duplicate (keep lowest energy).
- Optional:
      * --symm-rmsd / --symm-reflection (informational symmetry-aware RMSD via RDKit)
      * --no-shape to skip shape Tanimoto
      * --no-pair-log to suppress large per-pair report
"""

import os
import re
import argparse
import warnings
from collections import defaultdict, Counter
import sys

import numpy as np
from ase import Atoms
import cclib
from openbabel import pybel, openbabel  # noqa: F401

# Optional RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import rdShapeHelpers
    from rdkit.Chem import rdMolAlign
    from rdkit.Geometry import Point3D
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning, module="cclib")

# Constants / conversions
HARTREE_TO_KCAL = 627.509
EV_TO_HARTREE = 1.0 / 27.211386245988
CM1_TO_GHZ = 29.9792458

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05
}

THRESHOLDS = {
    'inertia': 1,        # amu * A^2
    'energy': 0.1,       # kcal/mol
    'dipole': 0.1,       # Debye
    'frequencies': 1.0,  # cm^-1
    'homo': 0.1,         # eV
    'lumo': 0.1,         # eV
    'rotconsts': 0.5,    # GHz
    'rmsd_low': 0.5,     # commentary threshold
    'rmsd_high': 1.0     # high RMSD threshold
}

# -------- Progress Bar --------
def print_progress_bar(current, total, bar_length=40):
    percent = current / total if total else 1
    filled_length = int(bar_length * percent)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\rProgress: |{bar}| {percent*100:5.1f}% ({current}/{total} pairs)")
    sys.stdout.flush()
    if current == total:
        print()  # Newline at end

# -------- Formula Helpers --------
def extract_stoichiometry(log_file):
    pat = re.compile(r'^\s*Stoichiometry\s+([A-Za-z0-9]+)\s*$')
    try:
        with open(log_file, 'r', errors='ignore') as fh:
            for line in fh:
                m = pat.match(line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None

def hill_formula_from_geometry(geometry):
    if not geometry:
        return "Unknown"
    counts = Counter(a for a, *_ in geometry)
    parts = []
    if 'C' in counts:
        c = counts.pop('C')
        parts.append(f"C{c if c>1 else ''}")
        if 'H' in counts:
            h = counts.pop('H')
            parts.append(f"H{h if h>1 else ''}")
        for elem in sorted(counts):
            n = counts[elem]
            parts.append(f"{elem}{n if n>1 else ''}")
    else:
        for elem in sorted(counts):
            n = counts[elem]
            parts.append(f"{elem}{n if n>1 else ''}")
    return ''.join(parts)

# -------- Parsing --------
def extract_gaussian_energy(log_file):
    scf_pattern = re.compile(r"^\s*SCF Done:\s+E\([RU]?[A-Za-z0-9]+\)\s*=\s*([-]?\d+\.\d+)")
    last_energy = None
    try:
        with open(log_file, "r", errors="ignore") as f:
            for line in f:
                m = scf_pattern.match(line)
                if m:
                    last_energy = float(m.group(1))
        if last_energy is not None:
            return last_energy
    except Exception as e:
        print(f"Warning: energy regex parse failed in {log_file}: {e}")

    try:
        data = cclib.io.ccread(log_file)
        if hasattr(data, "scfenergies") and data.scfenergies:
            return data.scfenergies[-1] * EV_TO_HARTREE
    except Exception:
        pass
    return None

def parse_quantum_properties(log_file):
    try:
        geometry = []
        pattern = re.compile(r'^\s*([A-Za-z]{1,3})\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s*$')
        in_zmatrix = False
        with open(log_file, 'r', errors="ignore") as fh:
            for line in fh:
                if 'Symbolic Z-matrix:' in line:
                    in_zmatrix = True
                    continue
                if in_zmatrix and ('Input orientation:' in line or line.strip() == ''):
                    in_zmatrix = False
                    break
                if in_zmatrix and 'Charge' not in line and 'Multiplicity' not in line:
                    m = pattern.match(line)
                    if m:
                        geometry.append((m.group(1),
                                         float(m.group(2)),
                                         float(m.group(3)),
                                         float(m.group(4))))
        energy = extract_gaussian_energy(log_file)
        data = cclib.io.ccread(log_file)
        props = {'energy': energy, 'geometry': geometry}

        if hasattr(data, 'moments') and len(data.moments) > 1:
            props['dipole'] = float(np.linalg.norm(data.moments[1]))
        else:
            props['dipole'] = None

        props['frequencies'] = np.sort(data.vibfreqs) if hasattr(data, 'vibfreqs') else None

        if hasattr(data, 'moenergies') and hasattr(data, 'homos'):
            try:
                homo_idx = data.homos[0]
                lumo_idx = homo_idx + 1
                moener = data.moenergies[0]
                props['homo'] = moener[homo_idx] if homo_idx < len(moener) else None
                props['lumo'] = moener[lumo_idx] if lumo_idx < len(moener) else None
            except Exception:
                props['homo'] = props['lumo'] = None
        else:
            props['homo'] = props['lumo'] = None

        if hasattr(data, 'rotconsts'):
            props['rotconsts'] = np.sort(data.rotconsts * CM1_TO_GHZ)
        else:
            props['rotconsts'] = None

        return props
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None

# -------- Geometry / RMSD --------
def get_inertia_moments(mol):
    mol.center()
    return np.sort(mol.get_moments_of_inertia())

def geometry_to_obmol(geometry):
    obmol = openbabel.OBMol()
    for atom, x, y, z in geometry:
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(openbabel.GetAtomicNum(atom))
        obatom.SetVector(x, y, z)
    return obmol

def obabel_rmsd(g1, g2):
    if not g1 or not g2:
        return None
    mol1 = geometry_to_obmol(g1)
    mol2 = geometry_to_obmol(g2)
    aligner = openbabel.OBAlign()
    aligner.SetRefMol(mol1)
    aligner.SetTargetMol(mol2)
    if aligner.Align():
        return aligner.GetRMSD()
    return None

def _bond_cutoff(sym1, sym2, scale=1.25):
    r1 = COVALENT_RADII.get(sym1, 0.75)
    r2 = COVALENT_RADII.get(sym2, 0.75)
    return scale * (r1 + r2)

def geometry_to_rdkit_mol(geometry, perceive_bonds=True, heavy_only=False):
    if not RDKit_AVAILABLE or not geometry:
        return None
    try:
        if heavy_only:
            geom_iter = [g for g in geometry if g[0].upper() != 'H'] or geometry
        else:
            geom_iter = geometry
        rw = Chem.RWMol()
        idx_map = []
        symbols = []
        for (sym, x, y, z) in geom_iter:
            rd_idx = rw.AddAtom(Chem.Atom(sym))
            idx_map.append((rd_idx, (x, y, z)))
            symbols.append(sym)
        if perceive_bonds:
            coords = [pos for _, pos in idx_map]
            n = len(coords)
            for i in range(n):
                xi, yi, zi = coords[i]
                for j in range(i+1, n):
                    xj, yj, zj = coords[j]
                    dx = xi - xj; dy = yi - yj; dz = zi - zj
                    if dx*dx + dy*dy + dz*dz <= _bond_cutoff(symbols[i], symbols[j])**2:
                        try:
                            rw.AddBond(i, j, Chem.BondType.SINGLE)
                        except Exception:
                            pass
        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for rd_idx, (x, y, z) in idx_map:
            conf.SetAtomPosition(rd_idx, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
        return mol
    except Exception:
        return None

def shape_tanimoto(mA, mB):
    if not RDKit_AVAILABLE or mA is None or mB is None:
        return None
    try:
        dist = rdShapeHelpers.ShapeTanimotoDist(mA, mB)
        return 1.0 - dist
    except Exception:
        return None

def symmetry_aware_rmsd(g1, g2, allow_reflection=False):
    if not RDKit_AVAILABLE:
        return None, None
    m1 = geometry_to_rdkit_mol(g1, perceive_bonds=True, heavy_only=True)
    m2 = geometry_to_rdkit_mol(g2, perceive_bonds=True, heavy_only=True)
    if m1 is None or m2 is None or m1.GetNumAtoms() != m2.GetNumAtoms():
        return None, None
    try:
        base = rdMolAlign.GetBestRMS(m1, m2)
    except Exception:
        return None, None
    refl_val = None
    if allow_reflection:
        try:
            m2_ref = Chem.Mol(m2)
            conf = m2_ref.GetConformer()
            for i in range(m2_ref.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, Point3D(-p.x, p.y, p.z))
            refl_val = rdMolAlign.GetBestRMS(m1, m2_ref)
            base = min(base, refl_val)
        except Exception:
            refl_val = None
    return base, refl_val

# -------- Duplicate Comparison --------
def compare_properties(props_list,
                       mols,
                       names,
                       group_name,
                       comparison_file,
                       shape_mols=None,
                       compute_shape=True,
                       do_symm=False,
                       do_symm_reflect=False,
                       geom_list=None,
                       write_pairs=True):
    n = len(props_list)
    duplicate_groups = [set([i]) for i in range(n)]
    HIGH_RMSD_CUTOFF = THRESHOLDS['rmsd_high']

    # Progress bar for pairwise comparisons
    total_pairs = n * (n - 1) // 2
    pair_count = 0
    PROGRESS_UPDATE_INTERVAL = 69  # update every 69 pairs

    f = None
    if write_pairs:
        f = open(comparison_file, 'w')
        f.write(f"=== Pairwise Comparison for {group_name} ===\n\n")

    def log(line):
        if write_pairs:
            f.write(line + "\n")

    for i in range(n):
        for j in range(i + 1, n):
            props1 = props_list[i]
            props2 = props_list[j]
            name_i = names[i]
            name_j = names[j]
            if write_pairs:
                log(f"{name_i} vs {name_j}:")
            is_dup = True
            reasons = []

            # Inertia
            inertia1 = get_inertia_moments(mols[i])
            inertia2 = get_inertia_moments(mols[j])
            inertia_diff = float(np.max(np.abs(inertia1 - inertia2)))
            if write_pairs:
                log(f"  Max inertia diff: {inertia_diff:.4f} amu·Å²")
            if inertia_diff > THRESHOLDS['inertia']:
                is_dup = False
                reasons.append(f"Inertia diff {inertia_diff:.4f}>{THRESHOLDS['inertia']}")

            if not props1 or not props2:
                if write_pairs:
                    log("  Missing quantum properties.")
                is_dup = False
                reasons.append("Missing quantum properties")
            else:
                # Energy
                if props1['energy'] is not None and props2['energy'] is not None:
                    ediff = abs(props1['energy'] - props2['energy']) * HARTREE_TO_KCAL
                    if write_pairs:
                        log(f"  Energy diff: {ediff:.6f} kcal/mol")
                    if ediff > THRESHOLDS['energy']:
                        is_dup = False
                        reasons.append(f"Energy diff {ediff:.6f}>{THRESHOLDS['energy']}")
                else:
                    if write_pairs:
                        log("  Energy not available")

                # Dipole
                if props1['dipole'] is not None and props2['dipole'] is not None:
                    ddiff = abs(props1['dipole'] - props2['dipole'])
                    if write_pairs:
                        log(f"  Dipole diff: {ddiff:.4f} Debye")
                    if ddiff > THRESHOLDS['dipole']:
                        is_dup = False
                        reasons.append(f"Dipole diff {ddiff:.4f}>{THRESHOLDS['dipole']}")
                elif write_pairs:
                    log("  Dipole not available")

                # Frequencies
                if props1['frequencies'] is not None and props2['frequencies'] is not None:
                    if len(props1['frequencies']) == len(props2['frequencies']):
                        fdiff = float(np.max(np.abs(props1['frequencies'] - props2['frequencies'])))
                        if write_pairs:
                            log(f"  Max frequency diff: {fdiff:.2f} cm^-1")
                        if fdiff > THRESHOLDS['frequencies']:
                            is_dup = False
                            reasons.append(f"Freq diff {fdiff:.2f}>{THRESHOLDS['frequencies']}")
                    else:
                        if write_pairs:
                            log("  Frequency comparison failed (size mismatch)")
                        is_dup = False
                        reasons.append("Frequency list mismatch")
                elif write_pairs:
                    log("  Frequencies not available")

                # HOMO / LUMO
                if props1['homo'] is not None and props2['homo'] is not None:
                    hdiff = abs(props1['homo'] - props2['homo'])
                    if write_pairs:
                        log(f"  HOMO diff: {hdiff:.6f} eV")
                    if hdiff > THRESHOLDS['homo']:
                        is_dup = False
                        reasons.append(f"HOMO diff {hdiff:.6f}>{THRESHOLDS['homo']}")
                elif write_pairs:
                    log("  HOMO not available")

                if props1['lumo'] is not None and props2['lumo'] is not None:
                    ldiff = abs(props1['lumo'] - props2['lumo'])
                    if write_pairs:
                        log(f"  LUMO diff: {ldiff:.6f} eV")
                    if ldiff > THRESHOLDS['lumo']:
                        is_dup = False
                        reasons.append(f"LUMO diff {ldiff:.6f}>{THRESHOLDS['lumo']}")
                elif write_pairs:
                    log("  LUMO not available")

                # Rot consts
                if props1['rotconsts'] is not None and props2['rotconsts'] is not None:
                    rdiff = float(np.max(np.abs(props1['rotconsts'] - props2['rotconsts'])))
                    if write_pairs:
                        log(f"  Max rot const diff: {rdiff:.4f} GHz")
                    if rdiff > THRESHOLDS['rotconsts']:
                        is_dup = False
                        reasons.append(f"Rot const diff {rdiff:.4f}>{THRESHOLDS['rotconsts']}")
                elif write_pairs:
                    log("  Rotational constants not available")

            property_reasons = list(reasons)

            # RMSD only if still candidate
            if is_dup and props1 and props2:
                try:
                    rmsd = obabel_rmsd(props1['geometry'], props2['geometry'])
                except Exception:
                    rmsd = None
                    is_dup = False
                    reasons.append("RMSD computation error")

                if write_pairs:
                    if rmsd is None:
                        log("  OpenBabel RMSD: could not compute")
                    else:
                        log(f"  OpenBabel RMSD: {rmsd:.3f} Å")

                if rmsd is not None:
                    if rmsd > THRESHOLDS['rmsd_high']:
                        if len(property_reasons) == 0:
                            if write_pairs:
                                log("  High RMSD but electronic properties identical -> treating as rotamer duplicates")
                        else:
                            if write_pairs:
                                log("  High RMSD: likely different stereochemistry")
                            is_dup = False
                            reasons.append(f"RMSD {rmsd:.3f}>{THRESHOLDS['rmsd_high']}")
                    elif rmsd < THRESHOLDS['rmsd_low']:
                        if write_pairs:
                            log("  Low RMSD: likely conformers")
                    else:
                        if write_pairs:
                            log("  Moderate RMSD: ambiguous")

            # Symmetry RMSD (informational)
            if do_symm and geom_list is not None:
                if props1 and props2:
                    sa_rmsd, refl_rmsd = symmetry_aware_rmsd(
                        geom_list[i], geom_list[j], allow_reflection=do_symm_reflect
                    )
                else:
                    sa_rmsd, refl_rmsd = (None, None)
                if write_pairs:
                    if sa_rmsd is not None:
                        log(f"  Symm RMSD (heavy atoms): {sa_rmsd:.3f} Å")
                        if do_symm_reflect and refl_rmsd is not None:
                            log(f"  Symm RMSD (reflected alt): {refl_rmsd:.3f} Å")
                            if refl_rmsd is not None and refl_rmsd < sa_rmsd - 1e-4:
                                log("  Note: Reflection lowers RMSD (possible enantiomer pair)")
                    else:
                        log("  Symm RMSD: not available")

            # Shape Tanimoto (informational)
            if compute_shape and shape_mols is not None and RDKit_AVAILABLE and write_pairs:
                molA = shape_mols[i]; molB = shape_mols[j]
                if molA is not None and molB is not None:
                    st = shape_tanimoto(molA, molB)
                    if st is not None:
                        log(f"  Shape Tanimoto: {st:.4f} (info only)")
                    else:
                        log("  Shape Tanimoto: n/a")
                else:
                    log("  Shape Tanimoto: skipped (missing mol)")
            elif compute_shape and not RDKit_AVAILABLE and write_pairs:
                log("  Shape Tanimoto: RDKit not available")

            if is_dup:
                if write_pairs:
                    log("  --> Likely duplicates\n")
                for group in duplicate_groups:
                    if i in group:
                        group.add(j)
                        break
            else:
                if write_pairs:
                    log("  --> Not duplicates. Reasons: " + ", ".join(reasons) + "\n")

            # Progress bar update
            pair_count += 1
            if pair_count % PROGRESS_UPDATE_INTERVAL == 0 or pair_count == total_pairs:
                print_progress_bar(pair_count, total_pairs)

    if write_pairs:
        f.close()

    # Merge overlapping sets
    unique_groups = []
    used = set()
    for idx in range(n):
        if idx in used:
            continue
        merged = set()
        for g in duplicate_groups:
            if idx in g:
                merged |= g
        unique_groups.append(merged)
        used |= merged

    # Lowest-energy representative
    unique_isomers = []
    for group in unique_groups:
        min_e = float('inf')
        min_idx = None
        for idx in group:
            p = props_list[idx]
            e = p['energy'] if p and p['energy'] is not None else None
            if e is not None and e < min_e:
                min_e = e
                min_idx = idx
        if min_idx is not None:
            unique_isomers.append(names[min_idx])
    return unique_isomers

# -------- Output Helpers --------
def save_unique_isomers(unique_isomers, props_list, names, group_name, output_file):
    with open(output_file, 'w') as f:
        f.write(f"Unique Conformers for {group_name}:\n")
        for conf in unique_isomers:
            idx = names.index(conf)
            props = props_list[idx]
            f.write(f"\nConformer: {conf}\n")
            f.write(f"Energy: {props['energy']:.6f} Hartrees\n" if props and props['energy'] is not None else "Energy: N/A\n")
            f.write(f"Dipole: {props['dipole']:.4f} Debye\n" if props and props['dipole'] is not None else "Dipole: N/A\n")
            f.write(f"HOMO: {props['homo']:.4f} eV\n" if props and props['homo'] is not None else "HOMO: N/A\n")
            f.write(f"LUMO: {props['lumo']:.4f} eV\n" if props and props['lumo'] is not None else "LUMO: N/A\n")
            f.write("Geometry:\n")
            if props and props.get('geometry'):
                for atom, x, y, z in props['geometry']:
                    f.write(f"{atom}\t{x:10.6f}\t{y:10.6f}\t{z:10.6f}\n")
            else:
                f.write("No geometry available\n")

def discover_log_basenames(log_dir):
    basenames = []
    for entry in os.listdir(log_dir):
        if entry.endswith(".log") and os.path.isfile(os.path.join(log_dir, entry)):
            basenames.append(entry[:-4])
    return sorted(basenames)

# -------- Main --------
def main():
    parser = argparse.ArgumentParser(
        description="Universal duplicate conformer detection (keeps original basenames)."
    )
    parser.add_argument("--logdir", required=True, help="Directory with Gaussian .log files.")
    parser.add_argument("--filelist", help="Optional file listing basenames (without .log).")
    parser.add_argument("--no-shape", action="store_true",
                        help="Disable RDKit shape similarity calculations.")
    parser.add_argument("--symm-rmsd", action="store_true",
                        help="Enable symmetry-aware RMSD (informational).")
    parser.add_argument("--symm-reflection", action="store_true",
                        help="Also test reflected coordinates (needs --symm-rmsd).")
    parser.add_argument("--no-pair-log", action="store_true",
                        help="Skip verbose per-pair comparison file (faster).")
    args = parser.parse_args()

    compute_shape = not args.no_shape
    do_symm = args.symm_rmsd
    do_symm_reflect = args.symm_reflection and do_symm
    write_pairs = not args.no_pair_log

    if (compute_shape or do_symm) and not RDKit_AVAILABLE:
        print("NOTE: RDKit not installed; shape & symmetry features skipped.")

    log_dir = os.path.abspath(args.logdir)
    if not os.path.isdir(log_dir):
        print(f"ERROR: log directory not found: {log_dir}")
        return

    # Collect basenames
    if args.filelist:
        if not os.path.isfile(args.filelist):
            print(f"ERROR: file list not found: {args.filelist}")
            return
        with open(args.filelist, 'r') as f:
            basenames = [ln.strip() for ln in f if ln.strip()]
        print(f"Using {len(basenames)} basenames from file list.")
    else:
        basenames = discover_log_basenames(log_dir)
        if not basenames:
            print("No .log files discovered.")
            return
        print(f"Auto-discovered {len(basenames)} .log files.")

    # Resolve paths & group by formula
    groups = defaultdict(list)
    missing = []
    for base in basenames:
        path = os.path.join(log_dir, base + ".log")
        if not os.path.exists(path):
            missing.append(base + ".log")
            continue
        props = parse_quantum_properties(path)
        geometry = props['geometry'] if props and props.get('geometry') else []
        formula = extract_stoichiometry(path)
        if formula is None:
            formula = hill_formula_from_geometry(geometry)
        groups[formula].append((path, base))

    if missing:
        print("Warning: Missing files:")
        for m in missing:
            print("  -", m)

    if not groups:
        print("No valid groups formed.")
        return

    # Map file (traceability)
    with open("original_name_map.txt", "w") as mf:
        mf.write("Formula\tOriginalBase\tLogFile\n")
        for formula, lst in groups.items():
            for path, base in lst:
                mf.write(f"{formula}\t{base}\t{path}\n")

    problematic = []
    all_unique = {}

    for formula, file_pairs in groups.items():
        props_list = []
        mols = []
        shape_mols = [] if (compute_shape and RDKit_AVAILABLE) else None
        geom_list = []
        names = []
        for (log_file, base) in sorted(file_pairs):
            props = parse_quantum_properties(log_file)
            if props is None or not props.get('geometry'):
                print(f"Problem parsing geometry for {base} ({log_file})")
                problematic.append(base)
            props_list.append(props)
            geometry = props['geometry'] if props and props.get('geometry') else []
            geom_list.append(geometry)
            symbols = [a for a, *_ in geometry]
            positions = [(x, y, z) for _, x, y, z in geometry]
            mols.append(Atoms(symbols=symbols, positions=positions))
            if shape_mols is not None:
                shape_mols.append(geometry_to_rdkit_mol(geometry, perceive_bonds=True, heavy_only=True))
            names.append(base)

        if mols:
            comparison_file = f"comparisons_{formula}.txt"
            uniques = compare_properties(
                props_list,
                mols,
                names,
                formula,
                comparison_file,
                shape_mols=shape_mols,
                compute_shape=compute_shape,
                do_symm=do_symm,
                do_symm_reflect=do_symm_reflect,
                geom_list=geom_list,
                write_pairs=write_pairs
            )
            save_unique_isomers(uniques, props_list, names, formula,
                                f"unique_isomers_{formula}.txt")
            all_unique[formula] = uniques

    # Combined list
    with open("unique_isomers_list_update.txt", "w") as f:
        f.write("Unique Conformers:\n")
        for formula, uni in all_unique.items():
            for conf in uni:
                f.write(f"{conf}\n")

    with open("problem_to_parse.txt", "w") as f:
        f.write("Problematic (parsing / geometry / RMSD issues):\n")
        for p in problematic:
            f.write(p + "\n")

    if not write_pairs:
        print("Done. Pairwise logs skipped (--no-pair-log).")
    else:
        print("Done.")

if __name__ == "__main__":
    main()
