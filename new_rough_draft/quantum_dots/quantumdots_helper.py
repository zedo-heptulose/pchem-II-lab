"""
Quantum Dots Lab Helper Module
==============================

This module contains visualization and calculation functions for the 
Quantum Dots and Quantum Yield lab. These functions support explorations of:
- Molecular Orbital Theory and tight-binding models
- Band Theory and semiconductor classification
- Fluorescence and Jablonski diagrams
- Machine Learning for quantum yield prediction

Functions are organized by lab section.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider


# =============================================================================
# PART 1: Molecular Orbital Theory Functions
# =============================================================================

def calculate_mo_energy_levels_tight_binding(n_atoms, interaction_strength=-1.0):
    """
    Calculate MO energy levels using tight binding model.
    
    This is a simplified model that demonstrates similar energy level patterns
    for a linear chain of atoms.
    
    Parameters
    ----------
    n_atoms : int
        Number of atoms in the chain
    interaction_strength : float
        Nearest-neighbor interaction strength (negative for bonding)
        
    Returns
    -------
    np.ndarray
        Sorted array of molecular orbital energies
    """
    # Atomic orbital energy (arbitrary zero)
    alpha = 0
    # Interaction strength (negative for bonding)
    beta = interaction_strength
    
    if n_atoms == 1:
        return np.array([alpha])
    
    # Calculate energies using tight binding model
    k_values = np.arange(1, n_atoms + 1)
    energies = alpha + 2 * beta * np.cos(k_values * np.pi / (n_atoms + 1))
    
    return np.sort(energies)


def plot_mo_diagram(mo_energies_list, atom_counts):
    """
    Plot MO diagram showing evolution with increasing atom count.
    
    Parameters
    ----------
    mo_energies_list : list of np.ndarray
        List of energy arrays for each atom count
    atom_counts : list of int
        Number of atoms corresponding to each energy array
    """
    plt.figure(figsize=(10, 8))
    
    for i, (mo_energies, n_atoms) in enumerate(zip(mo_energies_list, atom_counts)):
        x_positions = np.ones(len(mo_energies)) * i
        plt.scatter(x_positions, mo_energies, marker='_', s=500, 
                    color='blue', linewidth=2)
    
    plt.xticks(range(len(atom_counts)), [f"{n} atoms" for n in atom_counts])
    plt.ylabel("Energy (arbitrary units)")
    plt.title("Evolution of MO Energy Levels with Increasing Number of Atoms")
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.show()


def visualize_mo_schematic(n_atoms=4, selected_orbitals=None):
    """
    Generate schematic visualization of molecular orbitals.
    
    Creates simple schematic visualizations of the wavefunction patterns
    in a linear chain of atoms.
    
    Parameters
    ----------
    n_atoms : int
        Number of atoms in the chain
    selected_orbitals : list of int or None
        Which orbital indices to display (0-indexed)
    """
    if selected_orbitals is None:
        selected_orbitals = list(range(n_atoms))
    
    fig, axes = plt.subplots(len(selected_orbitals), 1, 
                             figsize=(8, 2 * len(selected_orbitals)))
    
    if len(selected_orbitals) == 1:
        axes = [axes]
    
    # Extend the range beyond the atoms to show full wavefunction
    x = np.linspace(-1, n_atoms, 200)
    
    for i, orbital_idx in enumerate(selected_orbitals):
        # The pattern for the nth molecular orbital in a linear chain
        # is approximated by a standing wave with n nodes
        n = orbital_idx + 1
        k = n * np.pi / (n_atoms + 1)
        psi = np.sin(k * (x + 1))
        
        # Plot the wavefunction
        axes[i].plot(x, psi, 'b-')
        axes[i].plot(x, -psi, 'b-')
        axes[i].fill_between(x, psi, 0, where=(psi > 0), color='blue', alpha=0.3)
        axes[i].fill_between(x, psi, 0, where=(psi < 0), color='blue', alpha=0.3)
        
        # Add atom positions
        atom_positions = np.arange(n_atoms)
        axes[i].plot(atom_positions, np.zeros_like(atom_positions), 'ko', ms=10)
        
        # Add orbital energy in a box
        energy = 2 * (-1.0) * np.cos(k)
        energy_text = f"E = {energy:.2f}"
        axes[i].text(n_atoms + 0.3, 0, energy_text, va='center',
                     bbox=dict(facecolor='white', edgecolor='black', 
                               boxstyle='round,pad=0.5'))
        
        # Label as bonding or antibonding
        bond_type = "Bonding" if orbital_idx < n_atoms / 2 else "Antibonding"
        
        axes[i].set_title(f"Molecular Orbital #{orbital_idx + 1} ({bond_type})")
        axes[i].set_ylim(-1.5, 1.5)
        axes[i].set_xlim(-1, n_atoms + 1)
        axes[i].set_yticks([])
        axes[i].set_xticks(atom_positions)
        axes[i].set_xticklabels([f"Atom {j + 1}" for j in range(n_atoms)])
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# PART 2: Band Theory Functions
# =============================================================================

def plot_interactive_band_structure(bandgap=1.5, valence_fill=1.0):
    """
    Interactive band structure showing bands with boxes.
    
    Parameters
    ----------
    bandgap : float
        Energy difference between bands (eV)
    valence_fill : float
        Fraction of valence band that is filled (0-1)
    """
    plt.figure(figsize=(8, 6))
    
    # Determine material type based on parameters
    if bandgap > 3.0:
        material_type = "insulator"
    elif bandgap < 0.1 or (valence_fill < 1.0):
        material_type = "conductor"
    else:
        material_type = "semiconductor"
    
    # Band width will be constant
    band_width = 1.0
    
    # Box dimensions
    box_width = 0.4
    box_x = [-box_width / 2, box_width / 2]
    
    # Energy ranges for bands
    core_band = [-band_width * 3 - 1, -band_width * 2 - 1]
    valence_band = [-band_width, 0]
    conduction_band = [bandgap, bandgap + band_width]
    
    # Calculate fill heights
    valence_fill_height = valence_band[0] + (valence_band[1] - valence_band[0]) * valence_fill
    
    # Draw core band (always fully filled)
    plt.fill_between(box_x, [core_band[0], core_band[0]], [core_band[1], core_band[1]], 
                     color='gold', edgecolor='black', linewidth=1.5, zorder=3)
    
    # Draw valence band - filled portion
    plt.fill_between(box_x, [valence_band[0], valence_band[0]], 
                     [valence_fill_height, valence_fill_height], 
                     color='gold', edgecolor=None, linewidth=0, zorder=3)
    
    # Draw valence band - unfilled portion
    if valence_fill < 1.0:
        plt.fill_between(box_x, [valence_fill_height, valence_fill_height], 
                         [valence_band[1], valence_band[1]], 
                         color='black', alpha=0.8, edgecolor=None, linewidth=0, zorder=3)
    
    # Draw valence band outline
    plt.plot([box_x[0], box_x[0]], [valence_band[0], valence_band[1]], 'k-', linewidth=1.5, zorder=4)
    plt.plot([box_x[1], box_x[1]], [valence_band[0], valence_band[1]], 'k-', linewidth=1.5, zorder=4)
    plt.plot([box_x[0], box_x[1]], [valence_band[0], valence_band[0]], 'k-', linewidth=1.5, zorder=4)
    plt.plot([box_x[0], box_x[1]], [valence_band[1], valence_band[1]], 'k-', linewidth=1.5, zorder=4)
    
    # Draw conduction band 
    conduction_fill = 0
    if material_type == "conductor" and bandgap < 0.1:
        conduction_fill = 1.0 - valence_fill
        conduction_fill_height = conduction_band[0] + (conduction_band[1] - conduction_band[0]) * conduction_fill
        
        plt.fill_between(box_x, [conduction_band[0], conduction_band[0]], 
                         [conduction_fill_height, conduction_fill_height], 
                         color='gold', edgecolor=None, linewidth=0, zorder=3)
        
        plt.fill_between(box_x, [conduction_fill_height, conduction_fill_height], 
                         [conduction_band[1], conduction_band[1]], 
                         color='black', alpha=0.8, edgecolor=None, linewidth=0, zorder=3)
    else:
        plt.fill_between(box_x, [conduction_band[0], conduction_band[0]], 
                         [conduction_band[1], conduction_band[1]], 
                         color='black', alpha=0.8, edgecolor=None, linewidth=0, zorder=3)
    
    # Draw conduction band outline
    plt.plot([box_x[0], box_x[0]], [conduction_band[0], conduction_band[1]], 'k-', linewidth=1.5, zorder=4)
    plt.plot([box_x[1], box_x[1]], [conduction_band[0], conduction_band[1]], 'k-', linewidth=1.5, zorder=4)
    plt.plot([box_x[0], box_x[1]], [conduction_band[0], conduction_band[0]], 'k-', linewidth=1.5, zorder=4)
    plt.plot([box_x[0], box_x[1]], [conduction_band[1], conduction_band[1]], 'k-', linewidth=1.5, zorder=4)
    
    # Add band labels
    plt.text(box_width * 0.7, core_band[0] + (core_band[1] - core_band[0]) / 2, 
             "Core\nBand", ha='left', va='center')
    plt.text(box_width * 0.7, valence_band[0] + (valence_band[1] - valence_band[0]) / 2, 
             "Valence\nBand", ha='left', va='center')
    plt.text(box_width * 0.7, conduction_band[0] + (conduction_band[1] - conduction_band[0]) / 2, 
             "Conduction\nBand", ha='left', va='center')
    
    # Label the bandgap if it exists
    if bandgap > 0.1:
        plt.annotate('', xy=(0, conduction_band[0]), xytext=(0, valence_band[1]),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        plt.text(box_width * 0.7, valence_band[1] + bandgap / 2, f"Bandgap\n{bandgap:.1f} eV", 
                 ha='left', va='center', color='red')
    
    # Calculate Fermi level
    if valence_fill < 1.0:
        fermi_level = valence_fill_height
    elif bandgap < 0.1:
        fermi_level = valence_band[1]
    else:
        fermi_level = valence_band[1] + bandgap / 2
    
    # Label Fermi level
    plt.axhline(y=fermi_level, color='blue', linestyle='--', alpha=0.7)
    plt.text(-box_width, fermi_level, "Fermi\nLevel", ha='right', va='center', color='blue')
    
    # Set labels and title
    plt.title(f"{material_type.capitalize()} Band Structure\n"
              f"Bandgap: {bandgap:.2f} eV, Valence Fill: {valence_fill * 100:.0f}%")
    plt.ylabel("Energy (eV)")
    plt.xlim(-box_width * 2, box_width * 2)
    plt.ylim(core_band[0] - 0.5, conduction_band[1] + 0.5)
    
    # Hide x-axis
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Add legend
    plt.plot([], [], color='gold', linewidth=10, label='Occupied States')
    plt.plot([], [], color='black', alpha=0.8, linewidth=10, label='Unoccupied States')
    plt.legend(loc='upper right')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def interactive_band_structure():
    """Create interactive widgets to control band structure parameters."""
    interact(
        plot_interactive_band_structure,
        bandgap=FloatSlider(min=0.0, max=5.0, step=0.1, value=1.5, 
                            description='Bandgap (eV):'),
        valence_fill=FloatSlider(min=0.0, max=1.0, step=0.05, value=1.0, 
                                 description='Valence Fill:')
    )


# =============================================================================
# PART 3: Fluorescence Functions
# =============================================================================

def plot_fluorescence_process(energy_gap=2.5, laser_wavelength=400):
    """
    Interactive visualization of fluorescence with Jablonski diagram and spectra.
    
    Parameters
    ----------
    energy_gap : float
        Energy difference between ground and excited states (eV)
    laser_wavelength : float
        Wavelength of excitation laser (nm)
    """
    # Set static y-axis limits for Jablonski diagram
    y_min = -0.5
    y_max = 4.0
    
    # Energy levels
    ground_state = 0
    excited_state = energy_gap
    
    # Define min and max thresholds for vibrational energy levels
    vib_min_threshold = excited_state
    vib_max_threshold = excited_state + 0.3
    
    # Calculate the middle energy of the vibrational range
    vib_middle_energy = (vib_min_threshold + vib_max_threshold) / 2
    
    # Calculate wavelength corresponding to the energies at the vibrational range boundaries
    min_absorption_wavelength = 1240 / (vib_max_threshold - ground_state)
    max_absorption_wavelength = 1240 / (vib_min_threshold - ground_state)
    
    # Use the average for the absorption center
    absorption_center = (min_absorption_wavelength + max_absorption_wavelength) / 2
    
    # Standard deviation for Gaussian
    absorption_width_parameter = (max_absorption_wavelength - min_absorption_wavelength) / 6
    
    # Calculate emission wavelength from energy gap (exaggerated Stokes shift)
    emission_center = 1240 / energy_gap + 50
    
    # Laser wavelength in eV
    laser_energy = 1240 / laser_wavelength
    
    # Determine where laser hits on energy scale
    target_energy = ground_state + laser_energy
    
    # Determine if laser energy is within the vibrational energy range
    laser_in_range = vib_min_threshold <= target_energy <= vib_max_threshold
    
    # Calculate emission intensity based on whether laser is in range
    if laser_in_range:
        absorption_efficiency = 1.0 - abs(target_energy - vib_middle_energy) / (vib_max_threshold - vib_min_threshold)
        emission_intensity = absorption_efficiency
    else:
        emission_intensity = 0.0
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # -----------------------
    # Jablonski Diagram (left subplot)
    # -----------------------
    
    # Create vibrational energy levels
    vib_count = 6
    ground_vib_levels = np.linspace(ground_state + 0.05, ground_state + 0.3, vib_count)
    excited_vib_levels = np.linspace(vib_min_threshold, vib_max_threshold, vib_count)
    
    # Draw energy levels (main electronic states as thicker lines)
    ax1.hlines(ground_state, 1, 3, linewidth=2, color='blue')
    ax1.hlines(excited_state, 1, 3, linewidth=2, color='blue')
    
    # Draw vibrational levels
    for vl in ground_vib_levels:
        ax1.hlines(vl, 1.1, 2.9, linewidth=1, color='blue', alpha=0.5)
    
    for vl in excited_vib_levels:
        ax1.hlines(vl, 1.1, 2.9, linewidth=1, color='blue', alpha=0.5)
    
    # Gray out the Jablonski diagram if laser is out of range
    if not laser_in_range:
        ax1.axhspan(y_min, y_max, alpha=0.3, color='gray')
        
        if target_energy > vib_max_threshold:
            message = "Laser energy too high\nNo absorption"
        else:
            message = "Laser energy too low\nNo absorption"
        
        ax1.text(2, (ground_state + excited_state) / 2, message, 
                 ha='center', va='center', color='red', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Always draw the absorption arrow to exactly target_energy
    if laser_in_range:
        ax1.arrow(1.5, ground_state, 0, target_energy - ground_state, 
                  head_width=0.1, head_length=0.1, fc='purple', ec='purple', width=0.02)
        
        # Non-radiative relaxation from target to excited state baseline
        ax1.plot([2, 2.3], [target_energy, excited_state], 'k--', alpha=0.7)
        
        # Emission arrow
        ground_vib_level = ground_vib_levels[1]
        ax1.arrow(2.5, excited_state, 0, -(excited_state - ground_vib_level), 
                  head_width=0.1, head_length=0.1, fc='red', ec='red', width=0.02)
    else:
        ax1.arrow(1.5, ground_state, 0, target_energy - ground_state, 
                  head_width=0.1, head_length=0.1, fc='gray', ec='gray', width=0.02, 
                  alpha=0.5, linestyle='--')
    
    # Labels
    ax1.text(0.7, ground_state, "Ground State\nS0", va='center')
    ax1.text(0.7, excited_state, "Excited State\nS1", va='center')
    
    # Add labels for the vibrational energy range
    ax1.text(3.2, vib_min_threshold, f"{vib_min_threshold:.2f} eV", va='bottom', ha='left', fontsize=8)
    ax1.text(3.2, vib_max_threshold, f"{vib_max_threshold:.2f} eV", va='top', ha='left', fontsize=8)
    ax1.plot([3.0, 3.1, 3.1], [vib_min_threshold, vib_min_threshold, vib_max_threshold], 'k-', linewidth=0.5)
    ax1.plot([3.1, 3.1], [vib_min_threshold, vib_max_threshold], 'k-', linewidth=1.5)
    ax1.plot([3.0, 3.1], [vib_max_threshold, vib_max_threshold], 'k-', linewidth=0.5)
    
    # Annotations for processes
    if laser_in_range:
        ground_vib_level = ground_vib_levels[1]
        ax1.text(1.35, (ground_state + target_energy) / 2, "Absorption\n(fs)", 
                 color='purple', ha='right')
        ax1.text(2.15, (target_energy + excited_state) / 2, "Vibrational\nRelaxation\n(ps)", 
                 color='black', ha='center', alpha=0.7)
        ax1.text(2.7, (excited_state + ground_vib_level) / 2, "Emission\n(ns)", 
                 color='red', ha='left')
    
    ax1.set_xlim(0.5, 3.5)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title('Jablonski Diagram')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_xticks([])
    
    # -----------------------
    # Absorption/Emission Spectra (right subplot)
    # -----------------------
    
    wavelengths = np.linspace(300, 700, 500)
    
    # Create absorption spectrum
    gaussian_sigma = absorption_width_parameter
    absorption = np.exp(-(wavelengths - absorption_center)**2 / (2 * gaussian_sigma**2))
    
    # Create emission spectrum (broader, red-shifted)
    emission_width = 15
    emission = np.exp(-(wavelengths - emission_center)**2 / (2 * emission_width**2))
    
    # Fix emission band to not cross the laser line
    if laser_in_range and laser_wavelength < emission_center:
        cutoff_factor = np.ones_like(wavelengths)
        cutoff_idx = wavelengths < laser_wavelength
        cutoff_factor[cutoff_idx] = 0.2
        emission = emission * cutoff_factor
    
    # Scale emission by absorption efficiency
    emission = emission * emission_intensity
    
    # Plot absorption and emission
    ax2.plot(wavelengths, absorption, 'b-', label='Absorption')
    ax2.plot(wavelengths, emission, 'r-', label='Emission')
    
    # Add annotations showing wavelength range of absorption band
    ax2.axvline(x=min_absorption_wavelength, color='blue', linestyle='--', alpha=0.5)
    ax2.axvline(x=max_absorption_wavelength, color='blue', linestyle='--', alpha=0.5)
    ax2.text(min_absorption_wavelength, 0.2, f"{min_absorption_wavelength:.0f} nm", 
             rotation=90, color='blue', ha='right', va='bottom', fontsize=8)
    ax2.text(max_absorption_wavelength, 0.2, f"{max_absorption_wavelength:.0f} nm", 
             rotation=90, color='blue', ha='left', va='bottom', fontsize=8)
    
    # Plot laser line as vertical line
    ax2.axvline(x=laser_wavelength, color='purple', linestyle='-', linewidth=2, label='Laser')
    
    # Add text about laser
    if laser_in_range:
        ax2.text(laser_wavelength + 5, 0.5, f"Laser: {laser_wavelength} nm", 
                 color='purple', ha='left', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    else:
        ax2.text(laser_wavelength + 5, 0.5, f"Laser: {laser_wavelength} nm\n(No absorption)", 
                 color='gray', ha='left', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add disclaimer about scale
    ax2.text(500, 0.05, "Note: Stokes shift exaggerated for visualization.", 
             ha='center', va='bottom', style='italic', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Absorption and Emission Spectra')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(300, 700)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()


def interactive_fluorescence():
    """Create interactive widgets to control fluorescence visualization."""
    interact(
        plot_fluorescence_process,
        energy_gap=FloatSlider(min=1.8, max=3.5, step=0.1, value=2.5, 
                               description='Energy Gap (eV):'),
        laser_wavelength=FloatSlider(min=300, max=600, step=5, value=400, 
                                     description='Laser (nm):')
    )


# =============================================================================
# PART 4: Machine Learning Visualization Functions
# =============================================================================

def plot_feature_importances(importances, feature_names=None):
    """
    Plot feature importances from a trained model.
    
    Parameters
    ----------
    importances : array-like
        Feature importance values from model
    feature_names : list of str, optional
        Names for each feature
    """
    if feature_names is None:
        feature_names = ['Diameter (nm)', 'Bandgap (eV)', 'Shell Ratio']
    
    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, importances)
    plt.title('Feature Importance for Quantum Dot Classification')
    plt.ylabel('Importance')
    plt.ylim(0, 1)
    plt.show()


def plot_model_predictions(X_test, y_test, y_pred, accuracy):
    """
    Plot test points with their predicted classes.
    
    Parameters
    ----------
    X_test : np.ndarray
        Test feature data
    y_test : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    accuracy : float
        Model accuracy score
    """
    correct = y_test == y_pred
    
    plt.figure(figsize=(8, 6))
    
    # Correct predictions
    plt.scatter(X_test[correct & (y_test == 0), 0], X_test[correct & (y_test == 0), 1], 
                color='red', marker='o', s=80, label='Correct: Low QY', alpha=0.7)
    plt.scatter(X_test[correct & (y_test == 1), 0], X_test[correct & (y_test == 1), 1], 
                color='blue', marker='o', s=80, label='Correct: High QY', alpha=0.7)
    
    # Incorrect predictions
    plt.scatter(X_test[~correct & (y_test == 0), 0], X_test[~correct & (y_test == 0), 1], 
                color='red', marker='X', s=100, edgecolors='black', linewidth=1.5,
                label='Incorrect: Actually Low QY', alpha=0.7)
    plt.scatter(X_test[~correct & (y_test == 1), 0], X_test[~correct & (y_test == 1), 1], 
                color='blue', marker='X', s=100, edgecolors='black', linewidth=1.5,
                label='Incorrect: Actually High QY', alpha=0.7)
    
    plt.xlabel('Diameter (nm)')
    plt.ylabel('Bandgap (eV)')
    plt.title(f'Model Predictions vs Actual Classes (Accuracy: {accuracy:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None):
    """
    Plot a confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.metrics.confusion_matrix
    class_names : list of str, optional
        Names for each class
    """
    if class_names is None:
        class_names = ['Low QY', 'High QY']
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def generate_quantum_dot_data(n_samples=100, random_state=42):
    """
    Generate synthetic quantum dot data for classification.
    
    This creates a simplified dataset where quantum yield depends on:
    - Size (moderate sizes are better, 3-6 nm)
    - Bandgap (optimal range 2.0-3.0 eV)
    - Shell ratio (thicker shells reduce surface defects)
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (size, bandgap, shell_ratio)
    y : np.ndarray
        Binary labels (1 = high quantum yield, 0 = low)
    """
    np.random.seed(random_state)
    
    # Create features
    size = np.random.uniform(2, 10, n_samples)           # Diameter in nm
    bandgap = np.random.uniform(1.5, 3.5, n_samples)     # Bandgap in eV
    shell_ratio = np.random.uniform(0, 0.5, n_samples)   # Core/shell thickness ratio
    
    X = np.column_stack([size, bandgap, shell_ratio])
    
    # Generate target: good quantum yield = 1, otherwise = 0
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Good QY if size is moderate, bandgap is optimal, and shell is thick
        if (3 <= size[i] <= 6) and (2.0 <= bandgap[i] <= 3.0) and (shell_ratio[i] > 0.2):
            y[i] = 1
        # Add some randomness (15% noise)
        if np.random.random() < 0.15:
            y[i] = 1 - y[i]
    
    return X, y
