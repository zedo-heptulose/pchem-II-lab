import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.special import sph_harm
import matplotlib.cm as cm
import ipywidgets as widgets
import psi4
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import warnings
from scipy.stats import norm
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`scipy.special.sph_harm` is deprecated"
)

def plot_qho_numeric(x, psi, L, dx, omega=1.0, scaled=False):
    """
    Plot a discretized wavefunction in the harmonic oscillator potential,
    along with the action of T, V, and H on that wavefunction.

    Parameters
    ----------
    x : ndarray
        Interior grid points.
    psi : ndarray
        Wavefunction values on the grid (eigenvector).
    L : ndarray
        Laplacian matrix (finite-difference second derivative).
    dx : float
        Grid spacing.
    omega : float
        Harmonic oscillator frequency (default = 1).
    """

    # Potential on the grid
    V_x = 0.5 * omega**2 * x**2
    V = np.diag(V_x)

    # Operators (units: ħ = m = 1)
    T = -0.5 * L
    H = T + V

    # Operator actions
    T_psi = T @ psi
    V_psi = V @ psi
    H_psi = H @ psi

    # Plot setup
    plt.figure(figsize=(4.5, 3))

    plt.axhline(0, color='grey', linewidth=0.8)

    def _rescale(y):
        maxval = np.max(np.abs(y))
        return y if maxval == 0 else y / maxval

    V_plot = V_x / np.max(V_x) * 0.3   # scales to [0,0.3]

    if not scaled:
        plt.plot(x, V_plot, color='grey', label=r'$V(x)$ (scaled)') 
        plt.plot(x, psi, color='black', label=r'$\psi(x)$')
        plt.plot(x, T_psi, color='blue', alpha=0.8, label=r'$\hat{T}\psi(x)$')
        plt.plot(x, V_psi, color='orange', alpha=0.8, label=r'$V(x)\psi(x)$')
        plt.plot(x, H_psi, color='red', alpha=0.8, label=r'$\hat{H}\psi(x)$')

    else:
        plt.plot(x, V_plot, color='grey', label=r'$V(x)$ (scaled)')
        plt.plot(x, psi, color='black', label=r'$\psi(x)$')
        plt.plot(x, _rescale(T_psi), color='blue', alpha=0.8, label=r'$\hat{T}\psi(x)$ (scaled)')
        plt.plot(x, _rescale(V_psi), color='orange', alpha=0.8, label=r'$V(x)\psi(x)$ (scaled)')
        plt.plot(x, _rescale(H_psi), color='red', alpha=0.8, label=r'$\hat{H}\psi(x)$ (scaled)')
    
    plt.legend(loc='upper left', prop={'size': 8})
    plt.title('Wavefunction in Harmonic Oscillator Potential')
    plt.xlabel('Position (arbitrary units)')
    plt.ylabel('Amplitude (arbitrary units)')
    plt.tight_layout()
    plt.show()

def animate_diagonalization_demo():
    """
    Animation demo: matrices as coordinate transformations and why diagonalization simplifies them.

    This is NOT meant to be physically relevant. It is a pure linear-algebra visualization.

    What it shows:
    - Left panel: an arbitrary basis where the same linear map mixes x and y.
    - Right panel: the eigenvector basis where the map is diagonal and only scales axes.

    Usage (in a notebook):
        from helper import animate_diagonalization_demo
        animate_diagonalization_demo()
    """
    # A simple symmetric matrix (guarantees real eigenvectors, nice diagonalization)
    A = np.array([[7.0, 5.0],
                  [1.0, 3.0]])

    # Eigen-decomposition: A = Q Λ Q^T
    eigvals, Q = np.linalg.eigh(A)
    Lam = np.diag(eigvals)

    # Points to transform: a circle + a few spokes for visual reference
    t = np.linspace(0, 2*np.pi, 200)
    circle = np.vstack([np.cos(t), np.sin(t)])  # shape (2, N)
    spokes = []
    for ang in np.linspace(0, 2*np.pi, 9)[:-1]:
        r = np.linspace(0, 1.0, 50)
        spokes.append(np.vstack([r*np.cos(ang), r*np.sin(ang)]))
    pts = np.hstack([circle] + spokes)  # shape (2, M)

    # Linear interpolation between identity (no transform) and full transform
    # Left:  p -> ((1-s)I + sA) p  in the standard basis
    # Right: same transform but shown in eigen-coordinates: u = Q^T p, u -> ((1-s)I + sΛ) u
    I = np.eye(2)

    def blend(M, s):
        return (1.0 - s) * I + s * M

    # Figure setup
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    axs[0].set_title("Arbitrary basis: mixing occurs")
    axs[1].set_title("Eigenbasis: pure scaling")

    for ax in axs:
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("coordinate 1")
        ax.set_ylabel("coordinate 2")

    # Plot artists
    (sc_left,) = axs[0].plot([], [], marker='.', linestyle='None', markersize=3)
    (sc_right,) = axs[1].plot([], [], marker='.', linestyle='None', markersize=3)

    text_left = axs[0].text(0.02, 0.95, "", transform=axs[0].transAxes, va='top')
    text_right = axs[1].text(0.02, 0.95, "", transform=axs[1].transAxes, va='top')

    # Precompute eigen-coordinate version of points
    pts_eig = Q.T @ pts

    # Animation update
    frames = 100

    def update(frame):
        s = frame / (frames - 1)

        # Left: in standard coordinates
        M_left = blend(A, s)
        pts_left = M_left @ pts

        # Right: transform in eigen-coordinates (diagonal), then plot in eigen-coordinates
        M_right = blend(Lam, s)
        pts_right = M_right @ pts_eig

        sc_left.set_data(pts_left[0], pts_left[1])
        sc_right.set_data(pts_right[0], pts_right[1])

        text_left.set_text(
            "Applying A mixes coordinates:\n"
            "new x depends on old x and y"
        )
        text_right.set_text(
            "In eigenbasis, A is diagonal:\n"
            "each coordinate scales independently"
        )

        return sc_left, sc_right, text_left, text_right

    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    plt.close(fig)  # so notebooks display only the animation object

    return anim

def plot_rigid_rotor(J, m):
    '''
    Adapted from:  https://github.com/DalInar/schrodingers-snake
    '''
    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2 * np.pi, 100)
    
    Theta, Phi = np.meshgrid(thetas, phis)
    s_harm = sph_harm(m, J, Phi, Theta) 

    R = np.abs(s_harm)  
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    phase = np.angle(s_harm)
    phase_norm = (phase + np.pi) / (2 * np.pi) 

    cmap = cm.hsv
    colors = cmap(phase_norm) 
    
    fig = plt.figure(figsize=(9, 8)) 
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                           facecolors=colors, linewidth=0, antialiased=False, alpha=0.8)

    ax.set_box_aspect([1, 1, 1]) 
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    max_range = max(x_limits[1] - x_limits[0], 
                    y_limits[1] - y_limits[0], 
                    z_limits[1] - z_limits[0]) / 2
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6, pad=0.05, ax=ax)

    cbar.set_label("Phase")
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-1$', r'$-i$', r'$1$', r'$i$', r'$-1$'])

    energy = J * (J + 1) 
    title_text = (rf'Rigid Rotor Wavefunction ( J={J}, m={m} )')
    plt.figtext(0.75, 0.20, f'$E ={energy} \\frac{{\\hbar^2}}{{2I}}$', ha='center', fontsize=14,color='red',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.suptitle(title_text, fontsize=18, ha='center',y=0.85)

    fig = plt.gcf()
    fig.canvas.draw()  
    line = plt.Line2D([0.18, 0.82], [0.81 , 0.81 ], 
                      color='black', linewidth=1, transform=fig.transFigure, clip_on=False)
    fig.add_artist(line)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    plt.show()

def plot_spherical_vector(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, color='cyan', alpha=0.1, 
                    edgecolor='black', linewidth=0.5)

    ax.plot([0, x], [0, y], [0, z], color='red', linewidth=2, label='r')
    ax.scatter(x, y, z, color='red', s=10)
    
    r_phi = 0.5  
    phi_arc = np.linspace(0, phi, 100)
    x_phi_arc = r_phi * np.cos(phi_arc)
    y_phi_arc = r_phi * np.sin(phi_arc)
    z_phi_arc = np.zeros_like(phi_arc)
    ax.plot(x_phi_arc, y_phi_arc, z_phi_arc, color='blue', label=r'$\phi$')
    
    r_theta = 0.5  
    theta_arc = np.linspace(0, theta, 100)
    x_theta_arc = r_theta * np.sin(theta_arc) * np.cos(phi)
    y_theta_arc = r_theta * np.sin(theta_arc) * np.sin(phi)
    z_theta_arc = r_theta * np.cos(theta_arc)
    ax.plot(x_theta_arc, y_theta_arc, z_theta_arc, color='green', label=r'$\theta$')

    max_range = 1.5
    ax.plot([0, max_range], [0, 0], [0, 0], color='black', linestyle=':')
    ax.plot([0, 0], [0, max_range], [0, 0], color='black', linestyle=':')
    ax.plot([0, 0], [0, 0], [0, max_range], color='black', linestyle=':')
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.legend(loc=6)
    title_text = "Spherical Coordinates"
    plt.suptitle(title_text, fontsize=18, ha='center',y=0.95,x=0.525)

    fig = plt.gcf()
    fig.canvas.draw()  
    line = plt.Line2D([0.18, 0.84], [0.89 , 0.89 ], 
                      color='black', linewidth=1, transform=fig.transFigure, clip_on=False)
    fig.add_artist(line)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def interactive_spherical_plot():
    widgets.interact(plot_spherical_vector, 
         r=(0.6, 1.5, 0.05),
         theta=(0, np.pi, 0.05),
         phi=(0, 2*np.pi, 0.05))

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

def create_psi4_molecule(smiles_string):
    '''
    INPUT:
    xyz format molecule coordinates
    OUTPUT:
    psi4-compatible molecular geometry object
    '''
    xyz_block = xyz_from_smiles(smiles_string)
    xyz_lines = xyz_block.split('\n')
    psi_coords = "\n".join(["0 1"] + xyz_lines[2:])
    psi4_molecule = psi4.geometry(psi_coords)
    return psi4_molecule

def gaussian(x, mu, sigma, intensity):
    return intensity * norm.pdf(x, mu, sigma)
    
def plot_spectrum(frequencies,intensities,title='Simulated IR Spectrum'):
    x_range = np.linspace(400, 4000, 2000)  
    spectrum = np.zeros_like(x_range)
        
    sigma = 10
    for freq, inten in zip(np.real(frequencies), intensities):
        
        spectrum += gaussian(x_range, freq, sigma, inten)
    
    spectrum /= spectrum.max()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, spectrum, color='black')
    plt.fill_between(x_range, spectrum, alpha=0.3, color='gray')
    plt.gca().invert_xaxis()  #for convention of higher values on the left in IR spectra
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Transmittance (Relative)")
    plt.title(title)
    plt.show()
