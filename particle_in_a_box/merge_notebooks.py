"""
Merge split PIB part notebooks into combined student and worked notebooks.

Creates:
  pib_student_v2.ipynb  (combined student version)
  pib_worked.ipynb      (combined worked version)

from:
  pib_part0_warmup.ipynb          (shared - identical in both)
  pib_part1_solving_se.ipynb      (student)
  pib_part1_solving_se_worked.ipynb
  pib_part2_energy_levels.ipynb   (student)
  pib_part2_energy_levels_worked.ipynb
  pib_part3_comp_chem.ipynb       (student)
  pib_part3_comp_chem_worked.ipynb
  pib_part4_comparing_models.ipynb (student)
  pib_part4_comparing_models_worked.ipynb
"""

import json
import copy


def load_nb(path):
    with open(path) as f:
        return json.load(f)


def save_nb(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"  Wrote {path}: {len(nb['cells'])} cells")


def strip_outputs(cell):
    """Remove outputs and reset execution count for a clean notebook."""
    cell = copy.deepcopy(cell)
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell


def make_title_cell():
    """Create the title/header markdown cell."""
    return {
        "cell_type": "markdown",
        "id": "afddfecc",
        "metadata": {},
        "source": [
            "# Particle in a Box and Molecular Conjugation\n",
            "\n",
            "In this lab, we will build, test, and compare mathematical models that describe electrons confined within molecules. Starting from physical principles, we will construct a particle-in-a-box model, solve it numerically, and compare its predictions to full quantum-chemical calculations on real molecules. Along the way, you will practice curve fitting, numerical optimization, and computational chemistry \u2014 skills that are central to modern physical chemistry research.\n",
            "\n",
            "---\n",
            "\n",
            "**Chemistry Learning Objectives:**\n",
            "- C4.1: Solve the Schr\u00f6dinger equation for the particle in a box using numerical methods.\n",
            "- C4.2: Connect features of PIB wavefunctions to their energies.\n",
            "- C4.3: Use the Self-Consistent Field (Hartree-Fock) method to predict chemical phenomena.\n",
            "- C4.4: Compare predictions of PIB and SCF models for conjugated polyenes.\n",
            "\n",
            "**Programming Learning Objectives:**\n",
            "- P4.1: Perform numerical optimization routines with `scipy`.\n",
            "- P4.2: Fit nonlinear curves to data using numerical optimization.\n",
            "- P4.3: Perform quantum chemical calculations with the `psi4` software package.\n",
            "\n",
            "---\n",
            "\n",
            "**Table of Contents**\n",
            "- [Warmup \u2014 Curve Fitting](#warmup)\n",
            "- [Part 1 \u2014 Solving the Schr\u00f6dinger Equation](#part1)\n",
            "- [Part 2 \u2014 Energy Levels of the Particle in a Box](#part2)\n",
            "- [Part 3 \u2014 Computational Chemistry](#part3)\n",
            "- [Part 4 \u2014 Comparing Models](#part4)\n",
            "- [Reflection](#reflection)"
        ]
    }


def make_imports_cell():
    """Create the consolidated imports cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "consolidated-imports",
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from scipy.optimize import curve_fit, minimize\n",
            "import psi4\n",
            "import fortecubeview\n",
            "import shutil\n",
            "\n",
            "from pib_helper import (compute_energy, create_psi4_molecule,\n",
            "                        load_psi4_molecule, calculate_r_squared)"
        ]
    }


def get_body_cells(nb, skip_title=False):
    """Get all cells from a part notebook, skipping the imports cell."""
    cells = nb['cells']
    body = []
    for cell in cells:
        src = cell_source(cell)
        # Skip import cells (first code cell in each part)
        if cell['cell_type'] == 'code' and (
            src.startswith('import numpy') or
            src.startswith('import np')
        ):
            continue
        # Skip the title cell from part0 (we create our own)
        if skip_title and cell['cell_type'] == 'markdown' and src.startswith('# Particle in a Box'):
            continue
        body.append(cell)
    return body


def cell_source(cell):
    """Get source as a single string."""
    src = cell.get('source', '')
    if isinstance(src, list):
        return ''.join(src)
    return src


def merge(part_files, output_path):
    """Merge part notebooks into one combined notebook."""
    # Start with title and imports
    cells = [make_title_cell(), make_imports_cell()]

    for i, path in enumerate(part_files):
        nb = load_nb(path)
        body = get_body_cells(nb, skip_title=(i == 0))
        for cell in body:
            cells.append(strip_outputs(cell))

    # Build the combined notebook
    combined = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "chm4411l",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    save_nb(combined, output_path)
    return combined


def main():
    student_parts = [
        'pib_part0_warmup.ipynb',
        'pib_part1_solving_se.ipynb',
        'pib_part2_energy_levels.ipynb',
        'pib_part3_comp_chem.ipynb',
        'pib_part4_comparing_models.ipynb',
    ]

    worked_parts = [
        'pib_part0_warmup.ipynb',
        'pib_part1_solving_se_worked.ipynb',
        'pib_part2_energy_levels_worked.ipynb',
        'pib_part3_comp_chem_worked.ipynb',
        'pib_part4_comparing_models_worked.ipynb',
    ]

    print("Merging student version...")
    merge(student_parts, 'pib_student_v2.ipynb')

    print("\nMerging worked version...")
    merge(worked_parts, 'pib_worked.ipynb')

    print("\nDone!")


if __name__ == '__main__':
    main()
