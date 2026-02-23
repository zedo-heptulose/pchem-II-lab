"""
Helper functions for the Particle in a Box lab.
Functions in this file support the notebook but are not
the focus of the learning activities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit


# ---------------------------------------------------------------------------
# Warmup: curve fitting utility
# ---------------------------------------------------------------------------

def fit_function(my_function, x_data, y_data, guess_parameters):
    """
    Fit a user-defined function to x, y data and plot the result.

    Parameters
    ----------
    my_function : callable
        Function of the form f(x, *params).
    x_data : array-like
        Independent variable data.
    y_data : array-like
        Dependent variable data.
    guess_parameters : list
        Initial guesses for the fit parameters.

    Returns
    -------
    params : ndarray
        Optimized parameters.
    """
    params, covariance = curve_fit(
        my_function, x_data, y_data, p0=guess_parameters
    )

    x_fit = np.linspace(np.min(x_data), np.max(x_data), 300)
    y_fit = my_function(x_fit, *params)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_data, y_data, color='red', label='Data', zorder=5, s=15)
    plt.plot(x_fit, y_fit, label='Fitted Function',
             color='green', linestyle='--', linewidth=2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curve Fit Result')
    plt.show()

    return params


# ---------------------------------------------------------------------------
# Part 1: energy evaluation and variational minimization
# ---------------------------------------------------------------------------

def compute_energy(psi, x):
    """
    Compute the energy expectation value <psi|H|psi>/<psi|psi>
    for a wavefunction on the grid x, in atomic units (hbar=1, m_e=1).

    Parameters
    ----------
    psi : ndarray
        Wavefunction values on the grid.
    x : ndarray
        Grid points (equally spaced).

    Returns
    -------
    E : float
        Energy expectation value in atomic units.
    """
    dx = x[1] - x[0]
    dpsi = np.diff(psi) / dx
    kinetic = 0.5 * np.sum(dpsi**2) * dx
    norm = np.sum(psi**2) * dx
    if norm < 1e-30:
        return np.inf
    energy = kinetic / norm
    # Penalty to enforce boundary conditions psi(0) = psi(L) = 0
    energy += 1e6 * (psi[0]**2 + psi[-1]**2)
    return energy


def run_energy_minimization(initial_guess_fn, L=1.0, N=200):
    """
    Minimize the energy expectation value <psi|H|psi>/<psi|psi>
    for a particle in a 1-D box of length L (atomic units).

    Parameters
    ----------
    initial_guess_fn : callable
        Function of x that returns an initial trial wavefunction.
        Must satisfy psi(0) = psi(L) = 0.
    L : float
        Length of the box (atomic units).  Default 1.0.
    N : int
        Number of grid points.

    Returns
    -------
    x : ndarray
        Grid points on [0, L].
    phi : ndarray
        Optimized, normalized wavefunction.
    E : float
        Converged energy expectation value (atomic units).
    """
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    phi0 = initial_guess_fn(x)
    phi0[0] = 0.0
    phi0[-1] = 0.0
    interior = phi0[1:-1].copy()

    def energy_functional(interior_vals):
        phi = np.zeros(N)
        phi[1:-1] = interior_vals
        dphi = np.diff(phi) / dx
        kinetic = 0.5 * np.sum(dphi**2) * dx
        norm = np.sum(phi**2) * dx
        if norm < 1e-30:
            return 1e10
        return kinetic / norm

    result = minimize(energy_functional, interior, method='L-BFGS-B',
                      options={'maxiter': 5000, 'ftol': 1e-14})

    phi_opt = np.zeros(N)
    phi_opt[1:-1] = result.x

    # Normalize
    norm = np.sqrt(np.sum(phi_opt**2) * dx)
    if norm > 0:
        phi_opt /= norm

    E = result.fun
    return x, phi_opt, E


# ---------------------------------------------------------------------------
# Part 3: computational chemistry utilities
# ---------------------------------------------------------------------------

def xyz_from_smiles(smiles_string):
    """
    Convert a SMILES string to an XYZ coordinate block.

    Parameters
    ----------
    smiles_string : str
        SMILES representation of a molecule.

    Returns
    -------
    xyz : str
        XYZ-format coordinate block.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        raise ValueError("Embedding failed for the molecule")
    result = AllChem.MMFFOptimizeMolecule(mol)
    if result != 0:
        raise ValueError("Optimization failed for the molecule")
    return Chem.MolToXYZBlock(mol)


def show_molecule(smiles_string):
    """
    Display a 3D visualization of a molecule from its SMILES string.

    Parameters
    ----------
    smiles_string : str
        SMILES representation of a molecule.
    """
    import py3Dmol

    xyz = xyz_from_smiles(smiles_string)
    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz, 'xyz')
    view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.2}})
    view.setStyle({'element': 'H'},
                  {'sphere': {'radius': 0.3, 'color': 'white'}})
    view.zoomTo()
    view.show()


def create_psi4_molecule(smiles_string):
    """
    Convert a SMILES string to a Psi4 molecule object.

    Parameters
    ----------
    smiles_string : str
        SMILES representation of a molecule.

    Returns
    -------
    psi4_molecule : psi4.core.Molecule
        Psi4-compatible molecule object.
    """
    import psi4

    xyz_block = xyz_from_smiles(smiles_string)
    xyz_lines = xyz_block.split('\n')
    psi_coords = "\n".join(["0 1"] + xyz_lines[2:])
    psi4_molecule = psi4.geometry(psi_coords)
    return psi4_molecule


def load_psi4_molecule(xyz_path):
    """
    Load a Psi4 molecule object from an XYZ file.

    Parameters
    ----------
    xyz_path : str
        Path to an XYZ-format file.

    Returns
    -------
    psi4_molecule : psi4.core.Molecule
        Psi4-compatible molecule object (neutral singlet).
    """
    import psi4

    with open(xyz_path) as f:
        xyz_lines = f.readlines()
    # Skip atom-count and comment lines; prepend charge/multiplicity and force C1
    coord_lines = [l.rstrip() for l in xyz_lines[2:] if l.strip()]
    psi_coords = "\n".join(["0 1"] + coord_lines + ["symmetry c1", "no_reorient", "no_com"])
    return psi4.geometry(psi_coords)


# ---------------------------------------------------------------------------
# Part 4: model comparison utilities
# ---------------------------------------------------------------------------

def calculate_r_squared(y_actual, y_predicted):
    """
    Compute the coefficient of determination (R-squared).

    Parameters
    ----------
    y_actual : array-like
        Observed values.
    y_predicted : array-like
        Model-predicted values.

    Returns
    -------
    r_squared : float
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)
    ss_residual = np.sum((y_actual - y_predicted)**2)
    return 1 - (ss_residual / ss_total)
