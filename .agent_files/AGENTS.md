# AGENTS.md

Guidelines for AI coding agents working in the PChem II Lab repository.

## Repository Overview

Jupyter notebook-based Physical Chemistry II laboratory course (CHM4411L). Contains computational chemistry and quantum mechanics experiments for undergraduate students.

**Project status**: Active revision for improved pedagogy and publication preparation.

## Environment Setup

```bash
# Create conda environment
conda env create -f chm4411l.yml

# Activate
conda activate chm4411l

# Launch notebooks
jupyter lab
```

### Key Dependencies
- Python 3.11.10
- Scientific: numpy, scipy, pandas, matplotlib, sympy
- Computational chemistry: psi4 (1.9.1), rdkit, py3dmol, fortecubeview
- Interactive: ipywidgets 8.1.2, jupyterlab 4.2.5

## Build/Test/Lint Commands

**No automated build system** - this is a notebook-based educational repository.

### Testing
- **Installation test**: Run `installation_test.ipynb` to verify environment
- **Notebook validation**: Execute notebooks top-to-bottom to verify they run without errors
- **No unit tests**: Manual verification required; execute cells sequentially

### Linting
No formal linting configuration. Follow code style guidelines below.

## Repository Structure

```
<module>/
├── *_student.ipynb    # Student version (incomplete code sections)
├── *_worked.ipynb     # Reference solution
├── *_helper.py        # Extracted helper functions (when present)
├── data/              # Experimental data (.csv, .xy)
└── experimental/      # Lab documentation (.docx)
```

### Lab Modules
diffraction/, magnetism/, nmr/, particle_in_a_box/, photoelectric/, 
quantum_dots/, ruby_laser/, vibration-rotation_hcl-dcl/, waves_and_optics/

### Key Documentation
- `CLAUDE.md` - Comprehensive agent instructions (primary reference)
- `TODO.md` - Revision checklist with specific bugs per notebook
- `AGENT_LOG.md` - Session continuity log
- `pchem_task_triage.md` - Task division (human vs AI)

## Code Style Guidelines

### Philosophy
- **Simplicity over performance**: Students are learning; prioritize readability
- **Zen of Python**: Explicit > implicit, simple > complex
- **Canonical patterns**: Use standard Python idioms that transfer to other contexts

### Imports
```python
# Standard library first, then third-party, grouped logically
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy import signal
```

### Formatting
- 4-space indentation (no tabs)
- Line length: ~100 characters max (flexible for notebooks)
- Spaces around operators: `x = a + b`, not `x=a+b`
- No trailing whitespace

### Naming Conventions
- Functions: `snake_case` - `calculate_distances()`, `plot_spectrum()`
- Variables: `snake_case` - `wavelength`, `cutoff_voltage`, `hkl_list`
- Constants: `UPPER_SNAKE_CASE` when appropriate
- Classes: `PascalCase` - `SuperpositionVisualizer`
- Descriptive names: `two_theta` not `tt`, `wavelength` not `w` (except in tight math loops)

### Functions
```python
def calculate_distances(two_theta, wavelength):
    """Brief description of what function does.
    
    Parameters
    ----------
    two_theta : array-like
        Diffraction angles in degrees.
    wavelength : float
        X-ray wavelength in Angstroms.
    
    Returns
    -------
    d : array
        Interplanar spacings.
    """
    two_theta = np.radians(two_theta)
    theta = two_theta / 2
    d = wavelength / (2 * np.sin(theta))
    return d
```

### Comments
- Explain *why*, not *what*
- Use subgoal labels in worked examples:
```python
# Subgoal: Load and inspect data
# Subgoal: Process/transform data
# Subgoal: Fit model to data
# Subgoal: Visualize and report
```

### Error Handling
- Provide informative error messages for student mistakes
- Use `raise ValueError()` with clear explanations
- Validate inputs where students might make errors:
```python
if starting_index > 10:
    raise ValueError('Could not converge on indices')
```

### Interactive Widgets Pattern
```python
def plot_function(param1, param2):
    # Plotting code
    plt.show()

def interactive_plot():
    widgets.interact(plot_function,
                     param1=(min, max, step),
                     param2=(min, max, step))
```

## Notebook Guidelines

### Structure
1. Library imports (top of notebook)
2. Warmup section (interactive exercises)
3. Main content (given functions → student code → visualizations → questions)
4. Reflection questions

### Cell Guidelines
- Keep code cells <15 lines, ONE purpose each
- Package imports at top of notebook
- Data files referenced with `data/` prefix: `pd.read_csv('data/file.csv')`

### Student Code Sections
Mark incomplete sections clearly:
```python
# YOUR CODE HERE
energy = # calculate the energy using the formula above
```

### Helper Module Pattern
Extract functions >10 lines that aren't pedagogically important:
```python
# In notebook:
from module_helper import *

# In module_helper.py: place visualization functions, data processing, etc.
```

## Data File Conventions

### Paths
Always use relative paths with `data/` prefix:
```python
# Correct
spectrum = pd.read_csv('data/KCl.csv')

# Wrong (missing data/)
spectrum = pd.read_csv('KCl.csv')
```

### Formats
- XRD data: `.xy` files, space-separated (2θ, intensity)
- Spectroscopic: `.csv` with headers
- Read XRD: `pd.read_csv('file.xy', sep=r'\s+', header=None, names=['2theta', 'intensity', '_'])`

## Computational Chemistry (Psi4)

### Standard Workflow
```python
# Subgoal: Define the molecule
smiles = "C=CC=CC=C"
psi4_molecule = create_psi4_molecule(smiles)

# Subgoal: Configure the calculation
psi4.set_output_file('output.dat', False)
psi4.set_memory('1 GB')

# Subgoal: Run and extract results
energy, wfn = psi4.energy('SCF/STO-3G', return_wfn=True, molecule=psi4_molecule)

# Subgoal: Visualize and interpret
# ... cube file generation and fortecubeview ...

# Always clean up
import shutil
shutil.rmtree('cubes')
```

### Atomic Units
- ℏ = 1, electron mass = 1
- 1 Angstrom = 1.8889 atomic units

## Known Issues

See `TODO.md` for comprehensive bug list. Common patterns:
- Wrong data paths (missing `data/` prefix)
- Undefined variables referenced before student code
- Column name mismatches in CSV files
- Typos in markdown cells

## Agent Task Boundaries

### Appropriate for AI agents
- Bug fixes (path corrections, typos, syntax errors)
- Code extraction to helper files
- Formatting standardization
- Adding subgoal labels to worked examples

### Requires human judgment
- Pedagogical content decisions
- Question quality assessment
- Difficulty calibration
- End-to-end notebook testing (requires Jupyter environment)

## References

- Primary guide: `CLAUDE.md` (416 lines of detailed instructions)
- Revision checklist: `TODO.md`
- Pedagogical framework: `pchem_literature_review_and_checklist.md`
