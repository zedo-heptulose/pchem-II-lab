"""
NMR Lab Helper Functions

This module contains visualization functions for NMR spectroscopy concepts.
These functions are used by nmr_student.ipynb and nmr_worked.ipynb.

Students should not need to modify these functions - they are provided for
interactive exploration of NMR concepts.

Functions:
    Visualization:
        plot_precession: Visualize spin precession in a magnetic field
        plot_fid: Visualize macroscopic magnetization and FID signal
        plot_nmr_visualization: Visualize FID and NMR spectrum with adjustable parameters
    
    Calculations (for student use):
        larmor_frequency: Calculate Larmor frequency in rad/s
        larmor_frequency_hz: Calculate Larmor frequency in Hz
        chemical_shift_ppm: Calculate chemical shift in ppm
        fwhm_from_T2: Calculate peak width from T2
        T2_from_fwhm: Calculate T2 from peak width
        generate_fid: Generate a synthetic FID signal
        fid_to_spectrum: Convert FID to frequency-domain spectrum via FFT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib import cm
from scipy.fft import fft, ifft, fftfreq


# =============================================================================
# Physical Constants
# =============================================================================

# Gyromagnetic ratios in rad/(s*T)
GAMMA_H = 2.6752218744e8    # 1H (proton)
GAMMA_C13 = 6.7282e7        # 13C
GAMMA_N15 = -2.712e7        # 15N
GAMMA_F19 = 2.5181e8        # 19F
GAMMA_P31 = 1.0839e8        # 31P

# Planck constants
HBAR = 1.054571817e-34      # Reduced Planck constant in J*s
PLANCK_H = 6.62607015e-34   # Planck constant in J*s


# =============================================================================
# Interactive Visualization Functions (GIVEN - do not modify)
# =============================================================================

def plot_precession(angle=30, B0=1.0, time=0.0):
    """
    Plot the precession of a spin in a magnetic field using atomic units.
    
    Parameters
    ----------
    angle : float
        Angle between the spin and the z-axis in degrees (0-180)
    B0 : float
        Strength of the magnetic field (relative scale, 0.2-2.0)
    time : float
        Time in arbitrary units (0-1 represents a full cycle)
    """
    # Clear previous plot
    clear_output(wait=True)
    
    # Create new figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # For simplicity, we'll use a scaled frequency that makes visualization easier
    # This makes one full time unit (0-1) correspond to one full cycle
    omega = 2 * np.pi * B0  # full cycle when time=1
    
    # Energy difference (proportional to B0)
    energy_diff = B0
    
    # Plot axes and labels
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Spin Precession in Magnetic Field\nB_0 = {B0:.1f}, Angle = {angle}')
    
    # Draw B0 field vector
    ax.quiver(0, 0, 0, 0, 0, 1.0, color='blue', arrow_length_ratio=0.1, label='B_0')
    
    # Current precession position 
    current_angle = omega * time
    x_position = np.sin(angle_rad) * np.cos(current_angle)
    y_position = np.sin(angle_rad) * np.sin(current_angle)
    z_position = np.cos(angle_rad)
    
    # Draw magnetic moment vector
    ax.quiver(0, 0, 0, x_position, y_position, z_position, 
             color='red', arrow_length_ratio=0.1, label='mu')
    
    # Draw the cone as a surface
    cone_half_angle = np.radians(20)  # Smaller cone for clarity
    
    # Generate cone using meshgrid for better surface plotting
    u = np.linspace(0, 1, 8)
    v = np.linspace(0, 2*np.pi, 12)
    
    U, V = np.meshgrid(u, v)
    
    # Create cone aligned with z-axis
    X = U * np.sin(cone_half_angle) * np.cos(V)
    Y = U * np.sin(cone_half_angle) * np.sin(V)
    Z = U * np.cos(cone_half_angle)
    
    # Get rotation angles
    phi = np.arctan2(y_position, x_position)
    theta = np.arccos(z_position)
    
    # Apply rotations to each point
    X_rot, Y_rot, Z_rot = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y, z = X[i,j], Y[i,j], Z[i,j]
            
            # Rotate around y-axis by theta
            x_new = x * np.cos(theta) + z * np.sin(theta)
            y_new = y
            z_new = -x * np.sin(theta) + z * np.cos(theta)
            
            # Rotate around z-axis by phi
            X_rot[i,j] = x_new * np.cos(phi) - y_new * np.sin(phi)
            Y_rot[i,j] = x_new * np.sin(phi) + y_new * np.cos(phi)
            Z_rot[i,j] = z_new
    
    # Plot the cone as a surface
    ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.3, color='green', 
                   rstride=1, cstride=1, linewidth=0)
    
    # Draw the circular path for precession
    theta_path = np.linspace(0, 2*np.pi, 40)
    x_path = np.sin(angle_rad) * np.cos(theta_path)
    y_path = np.sin(angle_rad) * np.sin(theta_path)
    z_path = np.cos(angle_rad) * np.ones_like(theta_path)
    ax.plot(x_path, y_path, z_path, 'b--', alpha=0.5, label='Precession path')
    
    # Current position on the precession path
    ax.scatter([x_position], [y_position], [z_position], color='red', s=50)
    
    # Add simplified formula display
    formula_text = (
        f"$\\langle I_x \\rangle = \\frac{{\\hbar}}{{2}}\\sin({angle})\\cos(\\omega t)$\n"
        f"$\\langle I_y \\rangle = -\\frac{{\\hbar}}{{2}}\\sin({angle})\\sin(\\omega t)$\n"
        f"$\\omega = \\gamma B_0 = {B0:.1f}$ (in relative units)"
    )
    plt.figtext(0.1, 0.05, formula_text)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_fid(B0=1.0, decay_rate=0.5, time=0.0):
    """
    Plot the macroscopic magnetization and FID signal.
    
    Parameters
    ----------
    B0 : float
        Strength of the magnetic field (relative scale, 0.2-2.0)
    decay_rate : float
        Rate of decay for the FID signal (0.2-2.0)
    time : float
        Current time (0-3 represents the duration of visualization)
    """
    time_points = 50
    # Clear previous plot
    clear_output(wait=True)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 0.5, 1])
    
    # 3D axis for magnetization (top left)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    # Text box for equations (top right)
    ax_text = fig.add_subplot(gs[0, -1])
    ax_text.axis('off')
    # 2D axis for FID (bottom)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Set up the magnetization plot
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_zlim(-0.2, 1.2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Macroscopic Magnetization')
    
    # Draw B0 field vector
    ax1.quiver(0, 0, 0, 0, 0, 1.0, color='blue', arrow_length_ratio=0.1, label='B_0')
    
    # Draw detection coil area vector (A)
    ax1.quiver(0, 0, 0, 1.0, 0, 0, color='green', arrow_length_ratio=0.1, label='A (coil)')
    
    # Calculate precession
    omega = 2 * np.pi * B0
    current_angle = omega * time
    
    # Magnetization decays exponentially
    magnitude = np.exp(-decay_rate * time)
    x_position = magnitude * np.cos(current_angle)
    y_position = magnitude * np.sin(current_angle)
    z_position = 0
    
    # Draw magnetization vector
    ax1.quiver(0, 0, 0, x_position, y_position, z_position, 
             color='red', arrow_length_ratio=0.1, label='M')
    
    # Draw precession path with decreasing radii
    radii = np.linspace(magnitude, 1.0, 4)[::-1]
    
    for r in radii:
        theta_path = np.linspace(0, 2*np.pi, 20)
        x_path = r * np.cos(theta_path)
        y_path = r * np.sin(theta_path)
        z_path = np.zeros_like(theta_path)
        alpha = 0.1 + 0.4 * (r/1.0)
        ax1.plot(x_path, y_path, z_path, 'b--', alpha=alpha)
    
    ax1.legend(loc='upper right')
    
    # Calculate FID signal
    t_array = np.linspace(0, 3, time_points)
    fid_signal = np.exp(-decay_rate * t_array) * np.sin(omega * t_array)
    
    current_index = min(int(time * (time_points - 1) / 3), time_points - 1)
    
    # Plot FID signal
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Voltage')
    ax2.set_title('Free Induction Decay (FID)')
    
    colors = ['black'] * time_points
    colors[current_index] = 'red'
    
    ax2.bar(t_array, fid_signal, width=2.4/time_points, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add formulas to text box
    formula_text = (
        f"$\\omega_0 = \\gamma B_0 = {B0:.1f}$\n\n"
        f"$\\Phi = M \\cdot A = M_x$\n\n"
        f"$V_{{\\mathrm{{induced}}}} = -\\frac{{d\\Phi}}{{dt}}$\n\n"
        f"$= -\\frac{{d(M_x)}}{{dt}}$\n\n"
        f"$\\propto e^{{-{decay_rate:.2f}t}} \\sin(\\omega_0 t)$"
    )
    ax_text.text(0.1, 0.5, formula_text, fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                 verticalalignment='center')
    
    plt.tight_layout()
    plt.show()


def plot_nmr_visualization(sigma=-0.0000005, T_2=1.0, T_1=1.0):
    """
    Plot FID and NMR spectrum with adjustable parameters.
    
    Parameters
    ----------
    sigma : float
        Shielding constant (0-15 ppm range for visualization)
    T_2 : float
        Transverse relaxation time (affects peak width, 0.1-2.0)
    T_1 : float
        Longitudinal relaxation time (affects peak intensity, 0.1-2.0)
    """
    # Clear previous output
    clear_output(wait=True)
    
    # Calculate derived parameters
    chemical_shift_ppm = sigma
    linewidth = 1/(T_2 * 10) 
    line_height = 1/T_1
    omega = (1-sigma) * 5
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 0.5, 1])
    
    # FID plot (top)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Spectrum plot (bottom)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Generate FID signal
    time_points = 1000
    t_array = np.linspace(0, 10, time_points)
    fid_signal = np.exp(-t_array/T_2) * np.sin(omega * t_array)
    
    # Plot FID
    ax1.bar(t_array, fid_signal, width=max(2.4/time_points, 0.01), color='black', alpha=0.7)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Voltage')
    ax1.set_title('Free Induction Decay (FID)')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Generate NMR spectrum
    def peak(x, chemical_shift, linewidth, line_height):
        return line_height * np.exp(-(x - chemical_shift)**2/(2 * linewidth**2))
    
    x = np.array([0])
    x = np.append(t_array, np.linspace(chemical_shift_ppm - linewidth * 4, 
                                        chemical_shift_ppm + linewidth * 4, 50))
    x = np.append(t_array, np.array([15]))
    y = peak(x, chemical_shift_ppm, linewidth, line_height)
    
    # Plot spectrum
    ax2.plot(x, y, 'r-')
    ax2.set_xlim(15, 0)  # Inverted x-axis (NMR convention)
    ax2.set_ylim(0, 5)
    ax2.set_xlabel('Chemical Shift (ppm)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('NMR Spectrum')
    ax2.grid(True, alpha=0.3)
    
    # Add formula for FID
    formula_text = (
        f"$\\sigma = {sigma:.2f}\\times{{}}10^{{-6}}$\n"
        f"$FID(t) = e^{{-t/T_2}} \\sin(\\omega t)$\n"
        f"$\\omega_0 = \\gamma(1-\\sigma)B_0$ \n"
        f"Chemical shift = $-\\sigma \\cdot 10^6$"
    )
    ax1.text(0.98, 0.95, formula_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("Note: This visualization uses a simplified model of NMR spectra for educational purposes.")


# =============================================================================
# Calculation Functions (for student use)
# =============================================================================

def larmor_frequency(gamma, B0, sigma=0):
    """
    Calculate the Larmor frequency for a nucleus in a magnetic field.
    
    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio in rad/(s*T)
    B0 : float
        Magnetic field strength in Tesla
    sigma : float, optional
        Shielding constant (dimensionless). Default is 0.
    
    Returns
    -------
    float
        Larmor frequency omega_0 in rad/s
    
    Notes
    -----
    The Larmor frequency is given by: omega_0 = gamma * (1 - sigma) * B0
    
    Examples
    --------
    >>> omega = larmor_frequency(GAMMA_H, 11.74)  # 500 MHz spectrometer
    >>> print(f"{omega:.2e} rad/s")
    """
    return gamma * (1 - sigma) * B0


def larmor_frequency_hz(gamma, B0, sigma=0):
    """
    Calculate the Larmor frequency in Hz.
    
    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio in rad/(s*T)
    B0 : float
        Magnetic field strength in Tesla
    sigma : float, optional
        Shielding constant (dimensionless). Default is 0.
    
    Returns
    -------
    float
        Larmor frequency nu_0 in Hz
    
    Examples
    --------
    >>> nu = larmor_frequency_hz(GAMMA_H, 11.74)
    >>> print(f"{nu/1e6:.1f} MHz")  # Should be ~500 MHz
    """
    omega = larmor_frequency(gamma, B0, sigma)
    return omega / (2 * np.pi)


def chemical_shift_ppm(nu_sample, nu_reference, nu_spectrometer):
    """
    Calculate chemical shift in parts per million (ppm).
    
    Parameters
    ----------
    nu_sample : float
        Resonance frequency of the sample nucleus in Hz
    nu_reference : float
        Resonance frequency of the reference (e.g., TMS) in Hz
    nu_spectrometer : float
        Operating frequency of the spectrometer in Hz
    
    Returns
    -------
    float
        Chemical shift in ppm
    
    Notes
    -----
    Chemical shift delta = (nu_sample - nu_reference) / nu_spectrometer * 10^6
    
    For 1H NMR, TMS (tetramethylsilane) is the standard reference with delta = 0 ppm.
    
    Examples
    --------
    >>> # A proton resonating 350 Hz downfield from TMS on a 500 MHz spectrometer
    >>> delta = chemical_shift_ppm(500e6 + 350, 500e6, 500e6)
    >>> print(f"{delta:.2f} ppm")  # Should be 0.70 ppm
    """
    return (nu_sample - nu_reference) / nu_spectrometer * 1e6


def fwhm_from_T2(T2):
    """
    Calculate the full width at half maximum (FWHM) of an NMR peak from T2.
    
    Parameters
    ----------
    T2 : float
        Transverse relaxation time in seconds
    
    Returns
    -------
    float
        FWHM in Hz
    
    Notes
    -----
    For a Lorentzian lineshape: FWHM = 1 / (pi * T2)
    
    This is a fundamental result from the Fourier transform of an 
    exponentially decaying sinusoid.
    
    Examples
    --------
    >>> fwhm = fwhm_from_T2(0.5)  # T2 = 500 ms
    >>> print(f"{fwhm:.2f} Hz")  # Should be ~0.64 Hz
    """
    return 1 / (np.pi * T2)


def T2_from_fwhm(fwhm):
    """
    Calculate T2 from the full width at half maximum (FWHM) of an NMR peak.
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum in Hz
    
    Returns
    -------
    float
        T2 relaxation time in seconds
    
    Examples
    --------
    >>> T2 = T2_from_fwhm(2.0)  # 2 Hz linewidth
    >>> print(f"{T2*1000:.1f} ms")  # Should be ~159 ms
    """
    return 1 / (np.pi * fwhm)


def generate_fid(frequencies, amplitudes, T2_values, t_max=1.0, sampling_rate=1000, 
                 add_noise=False, noise_level=0.05):
    """
    Generate a synthetic Free Induction Decay signal.
    
    Parameters
    ----------
    frequencies : array-like
        List of resonance frequencies in Hz
    amplitudes : array-like
        List of peak amplitudes
    T2_values : array-like
        List of T2 relaxation times in seconds
    t_max : float
        Maximum time in seconds
    sampling_rate : int
        Sampling rate in Hz
    add_noise : bool
        Whether to add Gaussian noise
    noise_level : float
        Standard deviation of noise (relative to signal amplitude)
    
    Returns
    -------
    t : ndarray
        Time array
    fid : ndarray
        Complex FID signal
    
    Examples
    --------
    >>> # Generate FID with two peaks at 50 Hz and 150 Hz
    >>> t, fid = generate_fid([50, 150], [1.0, 0.5], [0.5, 0.3])
    """
    n_points = int(t_max * sampling_rate)
    t = np.linspace(0, t_max, n_points)
    fid = np.zeros(n_points, dtype=complex)
    
    for freq, amp, T2 in zip(frequencies, amplitudes, T2_values):
        fid += amp * np.exp(-t / T2) * np.exp(2j * np.pi * freq * t)
    
    if add_noise:
        noise = noise_level * (np.random.randn(n_points) + 1j * np.random.randn(n_points))
        fid += noise
    
    return t, fid


def fid_to_spectrum(t, fid):
    """
    Convert a Free Induction Decay to a frequency-domain spectrum using FFT.
    
    Parameters
    ----------
    t : ndarray
        Time array in seconds
    fid : ndarray
        Complex FID signal
    
    Returns
    -------
    freq : ndarray
        Frequency array in Hz (positive frequencies only)
    spectrum : ndarray
        Magnitude spectrum (normalized)
    
    Examples
    --------
    >>> t, fid = generate_fid([50, 150], [1.0, 0.5], [0.5, 0.3])
    >>> freq, spectrum = fid_to_spectrum(t, fid)
    >>> plt.plot(freq, spectrum)
    >>> plt.xlabel('Frequency (Hz)')
    >>> plt.show()
    """
    N = len(fid)
    dt = t[1] - t[0]
    
    # Compute FFT
    fft_result = fft(fid)
    freq = fftfreq(N, dt)
    
    # Get positive frequencies only
    pos_mask = freq >= 0
    freq_pos = freq[pos_mask]
    spectrum_pos = np.abs(fft_result[pos_mask]) / N * 2  # Factor of 2 for one-sided
    
    return freq_pos, spectrum_pos
