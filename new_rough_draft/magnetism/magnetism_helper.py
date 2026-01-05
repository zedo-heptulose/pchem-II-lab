"""
Magnetism Lab Helper Functions
CHM4411L - Physical Chemistry II Laboratory

This module contains visualization functions for the magnetism lab.
Students should not modify these functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.special import sph_harm
import ipywidgets as widgets


class Arrow3D(FancyArrowPatch):
    """
    A 3D arrow for matplotlib 3D plots.
    
    Used to draw vectors representing angular momentum, magnetic dipole moments,
    and magnetic field directions in the vector model visualizations.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def zeeman_plot(l, ml, B):
    """
    Plot the orbital representation and vector model of angular momentum
    with Zeeman splitting in a magnetic field.
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number (l = 0, 1, 2, ...)
    ml : int
        Magnetic quantum number (-l <= ml <= l)
    B : float
        Magnetic field strength in Tesla
        
    Returns
    -------
    None
        Displays a matplotlib figure with two subplots:
        - Left: Spherical harmonic orbital representation
        - Right: Vector model showing L, Lz, and mu vectors
    """
    # Constants
    mu_B = 9.274e-24  # Bohr magneton in J/T
    hbar = 1.05457e-34  # Reduced Planck constant in J·s
    
    # Energy calculation for Zeeman splitting
    E_zeeman = -ml * mu_B * B
    mu_z = -ml * mu_B
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # === LEFT SUBPLOT: Orbital representation ===
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Calculate spherical harmonics
    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2 * np.pi, 100)
    Theta, Phi = np.meshgrid(thetas, phis)
    s_harm = sph_harm(ml, l, Phi, Theta)
    
    # Calculate radius and cartesian coordinates
    R = np.abs(s_harm)
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)
    
    # Phase coloring
    phase = np.angle(s_harm)
    phase_norm = (phase + np.pi) / (2 * np.pi)
    cmap = cm.hsv
    colors = cmap(phase_norm)
    
    # Plot orbital surface
    surf = ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                           facecolors=colors, linewidth=0, 
                           antialiased=False, alpha=0.8)
    
    # Add magnetic field direction    
    if B != 0:
        field_arrow = Arrow3D([0, 0], [0, 0], [0, B*0.1667], mutation_scale=20, 
                           lw=2, arrowstyle="-|>", color="blue")
        ax1.add_artist(field_arrow)
        ax1.text(0, 0, (B+0.2)*0.1667, "B", color='blue', fontsize=14, ha='center')
    
    # Set axis limits and labels
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_zlim(-0.5, 0.5)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'Orbital Representation (l={l}, ml={ml})')
    
    # === RIGHT SUBPLOT: Vector model representation ===
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Calculate angular momentum values
    L_tot = np.sqrt(l * (l + 1))
    L_z = ml
    
    # Calculate cone angle
    if L_tot > 0:
        theta_cone = np.arccos(L_z / L_tot)
    else:
        theta_cone = 0
    
    # Draw z-axis
    ax2.plot([0, 0], [0, 0], [-1.5, 1.5], 'k--', alpha=0.3)
    
    # Draw magnetic field direction
    if B != 0:
        field_arrow = Arrow3D([0, 0], [0, 0], [0, B], mutation_scale=20, 
                           lw=2, arrowstyle="-|>", color="blue")
        ax2.add_artist(field_arrow)
        ax2.text(0, 0, B+0.2, "B", color='blue', fontsize=14, ha='center')
    
    # Draw angular momentum vector
    if L_tot > 0:
        # Cone base circle
        phi = np.linspace(0, 2*np.pi, 100)
        L_xy = L_tot * np.sin(theta_cone)
        x_cone = L_xy * np.cos(phi)
        y_cone = L_xy * np.sin(phi)
        z_cone = np.ones_like(phi) * L_z
        ax2.plot(x_cone, y_cone, z_cone, 'r-', alpha=0.3)
        
        # Draw cone surface
        for phi_val in np.linspace(0, 2*np.pi, 20):
            x_line = [0, L_xy * np.cos(phi_val)]
            y_line = [0, L_xy * np.sin(phi_val)]
            z_line = [0, L_z]
            ax2.plot(x_line, y_line, z_line, 'k-', alpha=0.1)
        
        # Draw L vector (random position on the cone)
        phi_L = np.random.random() * 2 * np.pi
        L_x = L_xy * np.cos(phi_L)
        L_y = L_xy * np.sin(phi_L)
        L_arrow = Arrow3D([0, L_x], [0, L_y], [0, L_z],
                          mutation_scale=20, lw=2, arrowstyle="-|>", color="red")
        ax2.add_artist(L_arrow)
        
        Lz_arrow = Arrow3D([0, 0], [0, 0], [0, L_z],
                          mutation_scale=20, lw=2, arrowstyle="-|>", color="green")
        ax2.add_artist(Lz_arrow)
        
        ax2.text(L_x + 0.1, L_y + 0.1, L_z + 0.1, 
                r"$\vec{L}$", color='red', fontsize=14)
        ax2.text(0.1, 0.1, L_z + 0.1, 
                r"$L_z$", color='green', fontsize=14)    
        
        mu_arrow = Arrow3D([0, 0], [0, 0], [0, mu_z / mu_B], mutation_scale=20, lw=2, 
                       arrowstyle="-|>", color="purple")
        ax2.add_artist(mu_arrow)
        
        ax2.text(0, 0, mu_z/mu_B - 0.2, r"$\vec{\mu}$", color='purple', fontsize=14, ha='center')
    
    ax2.set_xlim(-3.0, 3.0)
    ax2.set_ylim(-3.0, 3.0)
    ax2.set_zlim(-3.0, 3.0)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Vector Model of Angular Momentum')
    
    energy_text = f'E = {E_zeeman:.3e} J'
    energy_box = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.figtext(0.5, 0.02, energy_text, ha='center', fontsize=14, 
               bbox=energy_box)
    
    info_text = [
        f'l = {l}',
        f'ml = {ml}',
        f'|L| = {L_tot:.2f}h',
        f'Lz = {L_z}h',
        f'$\\mu{{}}_z = {-L_z}\\mu_{{b}}$',
        f'B = {B} T'
    ]
    plt.figtext(0.49, 0.175, '\n'.join(info_text), va='center', ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle('Zeeman Effect: Orbital and Vector Representations', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout(pad=2)


def angular_momentum_coupling(l, ml, s, ms):
    """
    Plot the vector model of angular momentum coupling between orbital and spin.
    
    This visualization shows how orbital angular momentum (L) and spin angular
    momentum (S) combine to form total angular momentum (J).
    
    Parameters
    ----------
    l : float
        Orbital angular momentum quantum number
    ml : float
        Orbital magnetic quantum number
    s : float
        Spin quantum number (typically 0.5 for electron)
    ms : float
        Spin magnetic quantum number
        
    Returns
    -------
    None
        Displays a matplotlib 3D figure showing L, S, and J vectors
    """
    # Calculate magnitudes
    L_tot = np.sqrt(l * (l + 1))  # |L|
    L_z = ml                      # L_z
    S_tot = np.sqrt(s * (s + 1))  # |S|
    S_z = ms                      # S_z
    
    # Calculate J (total angular momentum)
    j = l + s  # For simplicity - can be adjusted for actual j value
    j_min = abs(l - s)
    j_max = l + s
    
    # Calculate J magnitudes
    J_tot = np.sqrt(j * (j + 1))  # |J|
    J_z = L_z + S_z               # J_z
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fixed position for L vector at 45 degrees in xy-plane
    L_xy = np.sqrt(max(0, L_tot**2 - L_z**2))  # magnitude of L in xy plane
    
    # Fix L at 45 degrees in xy-plane
    phi_L = np.pi/4  # 45 degrees
    L_x = L_xy * np.cos(phi_L)
    L_y = L_xy * np.sin(phi_L)
    
    # Draw z-axis
    ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'k--', alpha=0.3)
    
    # Draw orbital angular momentum vector
    L_arrow = Arrow3D([0, L_x], [0, L_y], [0, L_z],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color="blue")
    ax.add_artist(L_arrow)
    
    # Draw orbital angular momentum cone
    phi = np.linspace(0, 2*np.pi, 100)
    x_L_cone = L_xy * np.cos(phi)
    y_L_cone = L_xy * np.sin(phi)
    z_L_cone = np.ones_like(phi) * L_z
    ax.plot(x_L_cone, y_L_cone, z_L_cone, 'b-', alpha=0.3)
    
    # Draw L cone surface lines
    for phi_val in np.linspace(0, 2*np.pi, 16):
        x_line = [0, L_xy * np.cos(phi_val)]
        y_line = [0, L_xy * np.sin(phi_val)]
        z_line = [0, L_z]
        ax.plot(x_line, y_line, z_line, 'b-', alpha=0.1)
    
    # Draw L_z vector
    Lz_arrow = Arrow3D([0, 0], [0, 0], [0, L_z],
                       mutation_scale=15, lw=2, arrowstyle="-|>", color="cyan")
    ax.add_artist(Lz_arrow)
    
    # Position for spin vector - starts at tip of L vector
    # Determine if spin and orbital angular momentum are aligned or anti-aligned
    aligned = (L_z * S_z) >= 0
    theta_S = np.arccos(S_z / S_tot) if S_tot > 0 else 0
    
    # Choose phi_S based on alignment
    if aligned:
        phi_S = phi_L + np.pi
    else:
        phi_S = phi_L
    
    S_xy = S_tot * np.sin(theta_S)
    S_x = L_x + S_xy * np.cos(phi_S)
    S_y = L_y + S_xy * np.sin(phi_S)
    S_z_tip = L_z + S_z
    
    # Draw spin angular momentum vector
    S_arrow = Arrow3D([L_x, S_x], [L_y, S_y], [L_z, S_z_tip],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color="red")
    ax.add_artist(S_arrow)
    
    # Draw spin cone
    phi = np.linspace(0, 2*np.pi, 100)
    x_S_cone = L_x + S_xy * np.cos(phi)
    y_S_cone = L_y + S_xy * np.sin(phi)
    z_S_cone = np.ones_like(phi) * S_z_tip
    ax.plot(x_S_cone, y_S_cone, z_S_cone, 'r-', alpha=0.3)
    
    # Draw S cone surface lines
    for phi_val in np.linspace(0, 2*np.pi, 12):
        x_line = [L_x, L_x + S_xy * np.cos(phi_val)]
        y_line = [L_y, L_y + S_xy * np.sin(phi_val)]
        z_line = [L_z, S_z_tip]
        ax.plot(x_line, y_line, z_line, 'r-', alpha=0.1)
    
    # Draw total angular momentum vector J from origin to tip of S
    J_arrow = Arrow3D([0, S_x], [0, S_y], [0, S_z_tip],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color="green")
    ax.add_artist(J_arrow)
    
    # Add labels
    ax.text(L_x/2, L_y/2, L_z/2, "$\\vec{L}$", color='blue', fontsize=14)
    ax.text(L_x + S_xy*np.cos(phi_S)/2, L_y + S_xy*np.sin(phi_S)/2, 
            (L_z + S_z_tip)/2, "$\\vec{S}$", color='red', fontsize=14)
    ax.text(S_x/2, S_y/2, S_z_tip/2, "$\\vec{J}$", color='green', fontsize=14)
    ax.text(0, 0, L_z*0.9, "$L_z$", color='cyan', fontsize=12)
    ax.text(0, 0, 1.6, "$z$", fontsize=12)
    
    # Set axis limits and labels
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    
    # Add info box
    info_text = [
        f'l = {l}',
        f'$m_l = {ml}$',
        f'$L_z = {L_z}\\hbar$',
        f'$s = {s}$',
        f'$m_s = {ms}$',
        f'$Sz = {S_z}\\hbar$',
        f'$j = {j}$',
        f'$|J| = {J_tot:.2f}\\hbar$',
        f'$Jz = {J_z}\\hbar$'
    ]
    plt.figtext(0.225, 0.3, '\n'.join(info_text), va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Vector Model of Angular Momentum Coupling', fontsize=16)


def interactive_coupling_plot():
    """
    Create an interactive widget to explore angular momentum coupling.
    
    This function creates sliders for l, ml, s, and ms parameters and
    displays the angular momentum coupling visualization interactively.
    
    Returns
    -------
    None
        Displays an interactive widget with the angular_momentum_coupling plot
    """
    # Define the interactive controls
    l_widget = widgets.FloatSlider(min=0, max=3, step=1, value=2, description='l:')
    ml_widget = widgets.FloatSlider(min=-3, max=3, step=1, value=1, description='ml:')
    s_widget = widgets.FloatSlider(min=0.5, max=0.5, step=0.5, value=0.5, description='s:')
    ms_widget = widgets.FloatSlider(min=-0.5, max=0.5, step=1, value=0.5, description='ms:')
    
    # Create the interactive plot
    widgets.interact(angular_momentum_coupling, l=l_widget, ml=ml_widget, s=s_widget, ms=ms_widget)


# Psi4 helper functions for computational chemistry section
def xyz_from_smiles(smiles_string):
    """
    Convert a SMILES string to XYZ coordinate format.
    
    Parameters
    ----------
    smiles_string : str
        SMILES representation of a molecule
        
    Returns
    -------
    str
        XYZ format coordinate block
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
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
    """
    Display a 3D visualization of a molecule from its SMILES string.
    
    Parameters
    ----------
    smiles_string : str
        SMILES representation of a molecule
        
    Returns
    -------
    None
        Displays a py3Dmol interactive 3D view
    """
    import py3Dmol
    
    xyz = xyz_from_smiles(smiles_string)
    view = py3Dmol.view(width=200, height=200)
    view.addModel(xyz, 'xyz')
    view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.2}})
    view.setStyle({'element': 'H'}, {'sphere': {'radius': 0.3, 'color': 'white'}})
    view.zoomTo()
    view.show()


def create_psi4_molecule(smiles_string, charge=0, spin_multiplicity=1):
    """
    Create a Psi4 molecule object from a SMILES string.
    
    Parameters
    ----------
    smiles_string : str
        SMILES representation of a molecule
    charge : int, optional
        Molecular charge (default: 0)
    spin_multiplicity : int, optional
        Spin multiplicity: 1=singlet, 2=doublet, 3=triplet, etc. (default: 1)
        
    Returns
    -------
    psi4.core.Molecule
        Psi4-compatible molecular geometry object
    """
    import psi4
    
    xyz_block = xyz_from_smiles(smiles_string)
    xyz_lines = xyz_block.split('\n')
    psi_coords = "\n".join([f"{charge} {spin_multiplicity}"] + xyz_lines[2:])
    psi4_molecule = psi4.geometry(psi_coords)
    return psi4_molecule
