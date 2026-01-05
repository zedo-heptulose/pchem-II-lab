"""
Particle in a Box Lab Helper Functions
======================================

This module contains visualization and utility functions for the
Particle in a Box computational chemistry lab. These functions are
pre-implemented to allow students to focus on the core concepts.

Functions:
    - plot_wave: 3D visualization of PIB wavefunctions
    - interactive_pib_plot: Interactive widget for exploring k values
    - pib_energy_plot: 3D surface plot of PIB energy levels
    - xyz_from_smiles: Convert SMILES to XYZ coordinates
    - show_molecule: Display 3D molecular structure
    - create_psi4_molecule: Create Psi4 molecule object from SMILES
    - fit_function: Fit and plot arbitrary functions to data
    - get_gaps: Calculate HOMO-LUMO gaps for alkene series
    - calculate_r_squared: Compute R-squared for model fit
    - pib_fit_plot: Plot fitted PIB model against SCF data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from scipy.optimize import curve_fit

# Molecular visualization imports (optional - only needed for Psi4 sections)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import py3Dmol
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import psi4
    PSI4_AVAILABLE = True
except ImportError:
    PSI4_AVAILABLE = False


# =============================================================================
# WAVEFUNCTION VISUALIZATION
# =============================================================================

def plot_wave(k):
    """
    Plot the particle-in-a-box wavefunction as a 3D helix showing
    the real and imaginary components of the traveling wave solutions.
    
    The PIB wavefunction can be written as a superposition of left-
    and right-traveling waves: psi(x) = A*sin(kx) = A*(e^{ikx} - e^{-ikx})/(2i)
    
    Args:
        k (float): Wave vector (k = n*pi/L for allowed states)
    """
    x = np.linspace(0, 4, 400)
    exp_ikx = -1j * np.sqrt(1/2) * np.exp(1j * (k * x))
    exp_neg_ikx = 1j * np.sqrt(1/2) * np.exp(-1j * (k * x))
    sum_curve = exp_ikx + exp_neg_ikx

    box_mask = (x >= 0) & (x <= 4)
    
    real_part = np.real(exp_ikx) * box_mask
    imag_part = np.imag(exp_ikx) * box_mask
    real_part_neg = np.real(exp_neg_ikx) * box_mask
    imag_part_neg = np.imag(exp_neg_ikx) * box_mask
    real_part_sum = np.real(sum_curve) * box_mask
    imag_part_sum = np.imag(sum_curve) * box_mask
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[box_mask], imag_part[box_mask], real_part[box_mask], 
            label=r'$-ie^{ikx}$', alpha=0.6)
    ax.plot(x[box_mask], imag_part_neg[box_mask], real_part_neg[box_mask], 
            label=r'$ie^{-ikx}$', alpha=0.6)
    ax.plot(x[box_mask], imag_part_sum[box_mask], real_part_sum[box_mask], 
            label=r'$\sin(kx)$', linestyle='solid', color='black')

    # Plot boundary regions
    x_zero_left = np.linspace(-1, 0, 100)
    x_zero_right = np.linspace(4, 5, 100)
    ax.plot(x_zero_left, np.zeros_like(x_zero_left), np.zeros_like(x_zero_left),
            color='black', label='Boundary ($0 < x < 4$)')
    ax.plot(x_zero_right, np.zeros_like(x_zero_right), np.zeros_like(x_zero_right),
            color='black')

    ax.set_xlabel("x")
    ax.set_ylabel("Im components")
    ax.set_zlabel("Re components")
    ax.set_title("Particle in a Box Wavefunction")
    ax.legend()

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


def interactive_pib_plot():
    """
    Create an interactive plot for exploring which k values
    satisfy the particle-in-a-box boundary conditions.
    
    Use the slider to find values where psi(0) = psi(L) = 0.
    """
    widgets.interact(
        plot_wave, 
        k=widgets.FloatSlider(min=0.1, max=5, step=0.01, value=1,
                              description='k value:')
    )


def pib_energy_plot(pib_energy):
    """
    Create a 3D surface plot showing how PIB energy depends on
    both quantum number n and box length L.
    
    Args:
        pib_energy: Function that calculates PIB energy given (n, L_angs)
    """
    n_values = np.arange(1, 7) 
    L_values = np.arange(1, 7)  
    
    N, L = np.meshgrid(n_values, L_values)  
    E = np.vectorize(pib_energy)(N, L)  
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(N, L, E, cmap='viridis', edgecolor='k', 
                           alpha=0.7, norm=LogNorm())
    
    ax.set_xlabel("Quantum Number (n)")
    ax.set_ylabel("Box Length (L) [Ang]")
    ax.set_zlabel("Energy [a.u.]", labelpad=15)
    ax.set_title("Particle in a Box Energy Levels")
    
    ax.invert_yaxis()
    
    plt.show()


# =============================================================================
# MOLECULAR STRUCTURE FUNCTIONS
# =============================================================================

def xyz_from_smiles(smiles_string):
    """
    Convert a SMILES string to XYZ coordinates using RDKit.
    
    This function:
    1. Parses the SMILES string
    2. Adds explicit hydrogens
    3. Generates 3D coordinates
    4. Optimizes the geometry using MMFF force field
    
    Args:
        smiles_string (str): SMILES representation of molecule
        
    Returns:
        str: XYZ block with atomic coordinates
        
    Raises:
        ValueError: If embedding or optimization fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for this function")
        
    rdkit_molecule = Chem.MolFromSmiles(smiles_string)
    rdkit_molecule = Chem.AddHs(rdkit_molecule)
    result = AllChem.EmbedMolecule(rdkit_molecule, randomSeed=42)
    if result != 0:
        raise ValueError("Embedding failed for the molecule")
    result = AllChem.MMFFOptimizeMolecule(rdkit_molecule)
    
    if result != 0:
        raise ValueError("Optimization failed for the molecule")
    
    xyz = Chem.MolToXYZBlock(rdkit_molecule)
    return xyz


def show_molecule(smiles_string):
    """
    Display a 3D interactive view of a molecule from its SMILES string.
    
    Args:
        smiles_string (str): SMILES representation of molecule
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit and py3Dmol are required for this function")
        
    xyz = xyz_from_smiles(smiles_string)
    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz, 'xyz')
    view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.2}})
    view.setStyle({'element': 'H'}, {'sphere': {'radius': 0.3, 'color': 'white'}})
    view.zoomTo()
    view.show()


def create_psi4_molecule(smiles_string):
    """
    Create a Psi4-compatible molecule object from a SMILES string.
    
    Args:
        smiles_string (str): SMILES representation of molecule
        
    Returns:
        psi4.core.Molecule: Psi4 molecule object ready for calculations
    """
    if not PSI4_AVAILABLE:
        raise ImportError("Psi4 is required for this function")
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for this function")
        
    xyz_block = xyz_from_smiles(smiles_string)
    xyz_lines = xyz_block.split('\n')
    # Format: charge and multiplicity on first line, then coordinates
    psi_coords = "\n".join(["0 1"] + xyz_lines[2:])
    psi4_molecule = psi4.geometry(psi_coords)
    return psi4_molecule


# =============================================================================
# CURVE FITTING FUNCTIONS
# =============================================================================

def fit_function(my_function, guess_parameters, mystery_curve):
    """
    Fit an arbitrary function to the mystery curve data and display results.
    
    Args:
        my_function: Function with signature f(x, *params) -> y
        guess_parameters (list): Initial guesses for the parameters
        mystery_curve (DataFrame): Data with 'x' and 'y' columns
    """
    params, covariance = curve_fit(
        my_function,
        mystery_curve['x'], 
        mystery_curve['y'],
        p0=guess_parameters
    )
    
    x = mystery_curve['x']
    y_fitted = my_function(x, *params)
    
    plt.scatter(mystery_curve['x'], mystery_curve['y'], 
                color='red', label='Mystery Data')
    plt.plot(x, y_fitted, label='Fitted Function', 
             color='green', linestyle='--', linewidth=2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting arbitrary function to data')
    plt.show()
    
    print(f"Fitted parameters: {params}")


def get_gaps(n1, n2, quantum_energy_gap, pib_energy_gap):
    """
    Calculate HOMO-LUMO energy gaps for a series of alkenes.
    
    Args:
        n1 (int): Starting number of carbons (must be even)
        n2 (int): Ending number of carbons (must be even)
        quantum_energy_gap: Function to calculate SCF energy gap
        pib_energy_gap: Function to calculate PIB energy gap
        
    Returns:
        tuple: (quantum_gaps, pib_gaps) lists of energy gaps
    """
    vals = range(n1, n2 + 1, 2)
    quantum_gaps = []
    pib_gaps = []
    for n_c in vals:
        quantum_gaps.append(quantum_energy_gap(n_c))
        pib_gaps.append(pib_energy_gap(n_c))
    return quantum_gaps, pib_gaps


def calculate_r_squared(x, y, corrected_pib_model, coeffs):
    """
    Calculate R-squared value for a model fit.
    
    Args:
        x: Independent variable values
        y: Observed dependent variable values
        corrected_pib_model: Model function
        coeffs: Fitted coefficients
        
    Returns:
        float: R-squared value (1.0 = perfect fit)
    """
    y_pred = corrected_pib_model(x, *coeffs)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


def pib_fit_plot(x, y, params, corrected_pib_model):
    """
    Plot the parameterized PIB model against SCF data.
    
    Args:
        x: Chain lengths (number of carbons)
        y: SCF energy gaps
        params: Fitted parameters (alpha, beta)
        corrected_pib_model: Model function
    """
    x = np.array(range(4, 19, 2))
    y_fit = corrected_pib_model(x, *params)
    
    plt.scatter(x, y, label='SCF Data')
    r_squared = calculate_r_squared(x, y, corrected_pib_model, params)
    
    plt.plot(x, y_fit, color='red',
             label=f"Model: ${params[0]:.2f} \\times pib\\_energy\\_gap(n) + {params[1]:.2f}$ | $R^2 = {r_squared:.3f}$")
    plt.legend()
    plt.title('Parameterized PIB and SCF Data')
    plt.xlabel('Number of carbons')
    plt.ylabel('HOMO-LUMO Energy Gap [a.u.]')
    plt.show()
    
    print(f"Parameters: alpha = {params[0]:.4f}, beta = {params[1]:.4f}")
    print(f"R-squared: {r_squared:.4f}")
