import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import psi4

def xyz_from_smiles(smiles_string):
    rdkit_molecule = Chem.MolFromSmiles(smiles_string)
    rdkit_molecule = Chem.AddHs(rdkit_molecule)
    result = AllChem.EmbedMolecule(rdkit_molecule)
    if result != 0:
        raise ValueError("Embedding failed for the molecule")
    result = AllChem.MMFFOptimizeMolecule(rdkit_molecule)
    
    if result != 0:
        raise ValueError("Optimization failed for the molecule")
    
    xyz = Chem.MolToXYZBlock(rdkit_molecule)
    return xyz

def show_molecule(smiles_string):
    xyz = xyz_from_smiles(smiles_string)
    view = py3Dmol.view(width=200,height=200)
    view.addModel(xyz,'xyz')
    view.setStyle({'sphere':{'radius' : 0.3}, 'stick' : {'radius': 0.2}})
    view.setStyle({'element': 'H'}, {'sphere': {'radius': 0.3, 'color': 'white'}})
    view.zoomTo()
    view.show()

def create_psi4_molecule(smiles_string,charge=0,spin_multiplicity=1):
    '''
    INPUT:
    xyz format molecule coordinates
    OUTPUT:
    psi4-compatible molecular geometry object
    '''
    xyz_block = xyz_from_smiles(smiles_string)
    xyz_lines = xyz_block.split('\n')
    psi_coords = "\n".join([f"{charge} {spin_multiplicity}"] + xyz_lines[2:])
    psi4_molecule = psi4.geometry(psi_coords)
    return psi4_molecule


# --- Spectral simulation utilities ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaussian(x, mean, stdev, amplitude):
    """
    Parameters
    ----------
    x : array-like
        x values to evaluate over
    mean : float
        Center of the Gaussian
    stdev : float
        Standard deviation (width)
    amplitude : float
        Peak height

    Returns
    -------
    array-like
        Gaussian evaluated at each x
    """
    return amplitude * np.e**(-(x - mean)**2 / (2 * stdev**2))


def simulated_spectrum(transition_energies, xmin, xmax, npoints, amplitude=10):
    """
    Parameters
    ----------
    transition_energies : array-like
        List of transition energies (peak positions)
    xmin, xmax : float
        Range of the simulated spectrum
    npoints : int
        Number of points in the simulated spectrum
    amplitude : float
        Peak height

    Returns
    -------
    x, y : arrays
        Simulated spectrum
    """
    x = np.linspace(xmin, xmax, npoints)
    y = np.zeros_like(x)
    for energy in transition_energies:
        y += gaussian(x, energy, 0.01, amplitude)
    return x, y


def overlap(y1, y2):
    """
    Parameters
    ----------
    y1, y2 : array-like
        Two spectra (must correspond to the same x region)

    Returns
    -------
    float
        Normalized overlap between 0 and 1
    """
    return np.dot(y1, y2) / (1 + np.linalg.norm(y1) * np.linalg.norm(y2))


def get_spectral_params(peak_region):
    """
    Parameters
    ----------
    peak_region : pd.DataFrame
        DataFrame with 'wavenumber_cm-1' and 'intensity' columns

    Returns
    -------
    region_min, region_max, npoints, amp_max : float, float, int, float
    """
    region_min = peak_region['wavenumber_cm-1'].min()
    region_max = peak_region['wavenumber_cm-1'].max()
    npoints = len(peak_region)
    amp_max = peak_region['intensity'].max() / 2
    return region_min, region_max, npoints, amp_max


def spectra_overlap(transitions, peak_region):
    """
    Parameters
    ----------
    transitions : array-like
        List of transition energies
    peak_region : pd.DataFrame
        DataFrame with spectral region to compare predictions to

    Returns
    -------
    float
        Normalized overlap between simulated and experimental spectra
    """
    region_min, region_max, npoints, amp_max = get_spectral_params(peak_region)
    spec_x, spec_y = simulated_spectrum(transitions, region_min, region_max, npoints, amp_max)
    return overlap(spec_y, peak_region['intensity'].values)


def compare_spectra(transitions, peak_region):
    """
    Parameters
    ----------
    transitions : array-like
        List of transition energies
    peak_region : pd.DataFrame
        DataFrame with spectral region to compare
    """
    region_min, region_max, npoints, amp_max = get_spectral_params(peak_region)
    sim_x, sim_y = simulated_spectrum(transitions, region_min, region_max, npoints, amp_max)
    exp_x, exp_y = peak_region['wavenumber_cm-1'], peak_region['intensity']
    plt.plot(sim_x, sim_y, label='Simulated')
    plt.plot(exp_x, exp_y, alpha=0.5, linestyle='--', label='Experimental')
    plt.legend()
    plt.show()


