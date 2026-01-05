"""
Vibration-Rotation Lab Helper Functions
=======================================

This module contains visualization functions for the Vibration-Rotation
spectroscopy lab. Functions are extracted from the notebook to reduce
cognitive load and allow students to focus on the physics concepts.

Functions:
    - plot_qho: Plot wavefunction in quantum harmonic oscillator potential
    - plot_rigid_rotor: Plot spherical harmonic wavefunctions
    - plot_spherical_vector: Visualize spherical coordinates
    - interactive_spherical_plot: Interactive spherical coordinate explorer
    - gaussian: Gaussian function for spectrum broadening
    - plot_spectrum: Plot simulated IR spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sympy as sp
from scipy.special import sph_harm
from scipy.stats import norm
import ipywidgets as widgets


def plot_qho(wavefunction, x_symbol, omega_symbol):
    """
    Plot a wavefunction in the quantum harmonic oscillator potential.
    
    This function applies the Hamiltonian operator to the given wavefunction
    and plots both the original wavefunction and the result of H*psi.
    If the wavefunction is an eigenfunction, H*psi will be a scalar multiple
    of psi (the eigenvalue is the energy).
    
    Parameters
    ----------
    wavefunction : sympy expression
        The wavefunction to analyze, expressed in terms of x and omega
    x_symbol : sympy.Symbol
        The position symbol used in the wavefunction
    omega_symbol : sympy.Symbol
        The angular frequency symbol used in the wavefunction
    
    Returns
    -------
    None
        Displays the plot
    
    Example
    -------
    >>> x, omega = sp.symbols('x omega')
    >>> psi_0 = sp.exp(-omega * x**2 / 2)
    >>> plot_qho(psi_0, x, omega)
    """
    psi = sp.Symbol('psi')
    
    # Kinetic energy operator: T = -1/2 * d²/dx²  (in units where ℏ=m=1)
    T_psi = -sp.Rational(1, 2) * sp.Derivative(psi, x_symbol, 2)
    T_psi_eval = sp.lambdify(
        x_symbol,
        T_psi.subs(psi, wavefunction).doit().expand().subs(omega_symbol, 1)
    )
    
    # Potential energy: V = 1/2 * ω² * x²
    V = sp.Rational(1, 2) * omega_symbol**2 * x_symbol**2
    V_eval = sp.lambdify(x_symbol, V.subs(omega_symbol, 1))
    
    V_psi = V * psi
    V_psi_eval = sp.lambdify(
        x_symbol,
        V_psi.subs(psi, wavefunction).doit().expand().subs(omega_symbol, 1)
    )
    
    # Full Hamiltonian
    H_psi = T_psi + V_psi
    H_psi_subs = H_psi.subs(psi, wavefunction).doit().expand()
    H_psi_eval = sp.lambdify(
        x_symbol,
        H_psi.subs(psi, wavefunction).doit().expand().subs(omega_symbol, 1)
    )

    plt.figure(figsize=(4.5, 3))
    
    x_array = np.linspace(-4, 4, 100)
    
    # Zero line
    y_line = np.zeros_like(x_array)
    plt.plot(x_array, y_line, color='grey')
    
    # Potential energy curve
    potential_array = V_eval(x_array)
    plt.plot(x_array, potential_array, color='grey', label=r'$V(x)$')

    # Wavefunction
    wavefunction_eval = sp.lambdify(x_symbol, wavefunction.subs(omega_symbol, 1))
    y_wfn = wavefunction_eval(x_array)
    plt.plot(x_array, y_wfn, color='black', label=r'$\psi(x)$')
    
    # Hamiltonian applied to wavefunction
    y_operated_wfn = H_psi_eval(x_array)
    plt.plot(x_array, y_operated_wfn, color='red', alpha=0.8, label=r'$\hat{H}\psi(x)$')
    
    # Potential times wavefunction
    y_pot_wfn = V_psi_eval(x_array)
    plt.plot(x_array, y_pot_wfn, color='orange', alpha=0.8, label=r'$V(x)\psi(x)$')

    # Kinetic energy times wavefunction
    y_ke_wfn = T_psi_eval(x_array)
    plt.plot(x_array, y_ke_wfn, color='blue', alpha=0.8, label=r'$\hat{T}\psi(x)$')

    # Solve for eigenvalue
    _lambda = sp.Symbol('lambda')
    equation = sp.Eq(H_psi_subs, _lambda * wavefunction)
    solution = sp.solve(equation)
    
    if 'x' in str(solution[0][_lambda]):
        eigenvalue = 'n/a'
    else:
        eigenvalue = solution[0][_lambda]
        eigenvalue = str(eigenvalue).replace('omega', '\\omega').replace('*', '')
    plt.plot([], [], ' ', label=f"Energy: ${eigenvalue}$")
    
    plt.legend(loc=2, prop={'size': 8})
    plt.title('Wavefunctions in Harmonic Oscillator Potential')
    plt.ylabel('Amplitude (arbitrary units)')
    plt.xlabel('Position (arbitrary units)')
    plt.show()


def plot_rigid_rotor(J, m):
    """
    Plot the spherical harmonic wavefunction for a rigid rotor.
    
    This function visualizes the angular part of the rigid rotor wavefunction
    Y_J^m(theta, phi) as a 3D surface where the radius represents the 
    magnitude and the color represents the phase.
    
    Parameters
    ----------
    J : int
        The total angular momentum quantum number (J >= 0)
    m : int
        The magnetic quantum number (-J <= m <= J)
    
    Returns
    -------
    None
        Displays the plot
        
    Notes
    -----
    Adapted from: https://github.com/DalInar/schrodingers-snake
    """
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
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        facecolors=colors, linewidth=0, antialiased=False, alpha=0.8
    )

    ax.set_box_aspect([1, 1, 1])
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ) / 2
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
    title_text = rf'Rigid Rotor Wavefunction ( J={J}, m={m} )'
    plt.figtext(
        0.75, 0.20, f'$E ={energy} \\frac{{\\hbar^2}}{{2I}}$',
        ha='center', fontsize=14, color='red',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    plt.suptitle(title_text, fontsize=18, ha='center', y=0.85)

    fig = plt.gcf()
    fig.canvas.draw()
    line = plt.Line2D(
        [0.18, 0.82], [0.81, 0.81],
        color='black', linewidth=1, transform=fig.transFigure, clip_on=False
    )
    fig.add_artist(line)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    plt.show()


def plot_spherical_vector(r, theta, phi):
    """
    Visualize a vector in spherical coordinates.
    
    This function creates a 3D visualization showing a point at position
    (r, theta, phi) in spherical coordinates, along with the coordinate
    arcs illustrating the angles.
    
    Parameters
    ----------
    r : float
        The radial distance from origin
    theta : float
        The polar angle (from z-axis, 0 to pi)
    phi : float
        The azimuthal angle (in xy-plane, 0 to 2*pi)
    
    Returns
    -------
    None
        Displays the plot
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw transparent sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        xs, ys, zs, rstride=2, cstride=2, color='cyan', alpha=0.1,
        edgecolor='black', linewidth=0.5
    )

    # Draw position vector
    ax.plot([0, x], [0, y], [0, z], color='red', linewidth=2, label='r')
    ax.scatter(x, y, z, color='red', s=10)
    
    # Draw phi arc (azimuthal angle)
    r_phi = 0.5
    phi_arc = np.linspace(0, phi, 100)
    x_phi_arc = r_phi * np.cos(phi_arc)
    y_phi_arc = r_phi * np.sin(phi_arc)
    z_phi_arc = np.zeros_like(phi_arc)
    ax.plot(x_phi_arc, y_phi_arc, z_phi_arc, color='blue', label=r'$\phi$')
    
    # Draw theta arc (polar angle)
    r_theta = 0.5
    theta_arc = np.linspace(0, theta, 100)
    x_theta_arc = r_theta * np.sin(theta_arc) * np.cos(phi)
    y_theta_arc = r_theta * np.sin(theta_arc) * np.sin(phi)
    z_theta_arc = r_theta * np.cos(theta_arc)
    ax.plot(x_theta_arc, y_theta_arc, z_theta_arc, color='green', label=r'$\theta$')

    # Draw axes
    max_range = 1.5
    ax.plot([0, max_range], [0, 0], [0, 0], color='black', linestyle=':')
    ax.plot([0, 0], [0, max_range], [0, 0], color='black', linestyle=':')
    ax.plot([0, 0], [0, 0], [0, max_range], color='black', linestyle=':')
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.legend(loc=6)
    title_text = "Spherical Coordinates"
    plt.suptitle(title_text, fontsize=18, ha='center', y=0.95, x=0.525)

    fig = plt.gcf()
    fig.canvas.draw()
    line = plt.Line2D(
        [0.18, 0.84], [0.89, 0.89],
        color='black', linewidth=1, transform=fig.transFigure, clip_on=False
    )
    fig.add_artist(line)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


def interactive_spherical_plot():
    """
    Create an interactive spherical coordinate visualization.
    
    Uses ipywidgets to create sliders for r, theta, and phi that
    update the spherical coordinate visualization in real-time.
    
    Returns
    -------
    None
        Displays the interactive widget
    """
    widgets.interact(
        plot_spherical_vector,
        r=(0.6, 1.5, 0.05),
        theta=(0, np.pi, 0.05),
        phi=(0, 2*np.pi, 0.05)
    )


def gaussian(x, mu, sigma, intensity):
    """
    Compute a Gaussian peak for spectrum broadening.
    
    Parameters
    ----------
    x : array-like
        The x values (wavenumbers)
    mu : float
        Peak center (mean)
    sigma : float
        Peak width (standard deviation)
    intensity : float
        Peak intensity (height scaling)
    
    Returns
    -------
    array-like
        Gaussian peak values
    """
    return intensity * norm.pdf(x, mu, sigma)


def plot_spectrum(frequencies, intensities, title='Simulated IR Spectrum'):
    """
    Plot a simulated IR spectrum with Gaussian broadening.
    
    Takes discrete frequencies and intensities from a frequency calculation
    and broadens each peak with a Gaussian to simulate experimental line
    broadening effects.
    
    Parameters
    ----------
    frequencies : array-like
        Vibrational frequencies in cm^-1
    intensities : array-like
        IR intensities
    title : str, optional
        Plot title (default: 'Simulated IR Spectrum')
    
    Returns
    -------
    None
        Displays the plot
    
    Notes
    -----
    - Uses Gaussian broadening with sigma=10 cm^-1
    - X-axis is inverted to match IR spectroscopy convention
    - Spectrum is normalized to maximum intensity of 1
    """
    x_range = np.linspace(400, 4000, 2000)
    spectrum = np.zeros_like(x_range)
    
    sigma = 10
    for freq, inten in zip(np.real(frequencies), intensities):
        spectrum += gaussian(x_range, freq, sigma, inten)
    
    # Normalize
    if spectrum.max() > 0:
        spectrum /= spectrum.max()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, spectrum, color='black')
    plt.fill_between(x_range, spectrum, alpha=0.3, color='gray')
    plt.gca().invert_xaxis()  # Convention: higher wavenumbers on left
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Transmittance (Relative)")
    plt.title(title)
    plt.show()
