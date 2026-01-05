"""
Helper functions for the Waves and Optics lab.
This module contains visualization functions for wave superposition, Fourier transforms,
standing waves, and wave interference patterns.
"""

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from IPython.display import HTML


# =============================================================================
# WAVE SUPERPOSITION FUNCTIONS
# =============================================================================

def waves_and_superposition(f1, p1, f2, p2):
    """
    Generate two sine waves and their superposition.
    
    Parameters
    ----------
    f1 : float
        Frequency of first wave (Hz)
    p1 : float
        Phase of first wave (radians)
    f2 : float
        Frequency of second wave (Hz)
    p2 : float
        Phase of second wave (radians)
    
    Returns
    -------
    tuple
        (x, y_1, y_2, y_superposition) - time array and wave amplitudes
    """
    x = np.linspace(0, 10, 640)
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    y_1 = np.sin(w1 * x + p1)
    y_2 = np.sin(w2 * x + p2)
    y_superposition = y_1 + y_2
    return x, y_1, y_2, y_superposition


def fourier_transform(signal, sampling_rate=64):
    """
    Compute the Fourier transform of a signal.
    
    Parameters
    ----------
    signal : array-like
        Input signal
    sampling_rate : int
        Sampling rate in Hz
    
    Returns
    -------
    tuple
        (frequencies, magnitudes) - positive frequencies and FFT magnitudes
    """
    T = 1 / sampling_rate
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_signal = fft_signal / n
    frequencies = np.fft.fftfreq(n, T)
    positive_frequencies = frequencies[:n // 2]
    fft_signal_magnitude = np.abs(fft_signal[:n // 2])
    return positive_frequencies, fft_signal_magnitude


def superposition_plot(f1, p1, f2, p2):
    """
    Plot two waves, their superposition, and the Fourier transform.
    
    Parameters
    ----------
    f1, p1, f2, p2 : float
        Frequencies and phases for the two waves
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    x, y_1, y_2, y_superposition = waves_and_superposition(f1, p1, f2, p2)

    ax1.plot(x, y_1, label='First Wave', alpha=0.6, color='red')
    ax1.plot(x, y_2, label='Second Wave', alpha=0.6, color='orange')
    ax1.plot(x, y_superposition, label='Sum of waves', color='blue')
    ax1.set_title('Wave Superposition')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-3, 3)
    ax1.set_xlim(0, 4)
    ax1.legend()

    fourier_x, fourier_y = fourier_transform(y_superposition)
    ax2.plot(fourier_x, fourier_y)
    ax2.set_title('Fourier Transform of Resulting Wave')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Contribution')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 1.05)

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# RAYLEIGH QUOTIENT MINIMIZATION FUNCTIONS
# =============================================================================

def minimize_rayleigh(L, N, phi0_fn, rq_fn, method="L-BFGS-B", maxiter=200000, ftol=1e-12, plot=True):
    """
    Perform Rayleigh-quotient minimization for a 1D cavity with Dirichlet boundaries
    and return the converged standing-wave shape.

    Parameters
    ----------
    L : float
        Length of the domain.
    N : int
        Number of grid points for the discretization.
    phi0_fn : callable
        Function phi0_fn(x, L) -> array of shape (N,), initial guess satisfying phi(0)=phi(L)=0.
    rq_fn : callable
        Function rq_fn(phi, dx) -> float that computes the Rayleigh quotient.
    method : str, optional
        Optimization method (default "L-BFGS-B").
    maxiter : int, optional
        Maximum number of optimization iterations.
    ftol : float, optional
        Function tolerance for convergence.
    plot : bool, optional
        If True, show a plot of the converged mode.

    Returns
    -------
    x : ndarray of shape (N,)
        Grid positions.
    phi_star : ndarray of shape (N,)
        Normalized converged mode profile.
    """
    from scipy.optimize import minimize

    # Discretize domain
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    # Generate initial guess
    phi0 = np.asarray(phi0_fn(x, L), dtype=float)
    phi0[0] = 0.0
    phi0[-1] = 0.0

    # Optimization variables (interior only)
    u0 = phi0[1:-1].copy()

    def objective(u_vec):
        phi = np.zeros_like(phi0)
        phi[1:-1] = u_vec
        R = rq_fn(phi, dx)
        return R

    # Run minimization
    res = minimize(
        objective,
        u0,
        method=method,
        options={"maxiter": maxiter, "maxfun": maxiter, "ftol": ftol}
    )

    # Rebuild and normalize converged wave
    phi_star = np.zeros_like(phi0)
    phi_star[1:-1] = res.x
    phi_star[0] = 0.0
    phi_star[-1] = 0.0

    norm2 = np.sum(phi_star**2) * dx
    if norm2 > 0:
        phi_star /= np.sqrt(norm2)

    # Plot the result
    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(x, phi_star, lw=2, color='royalblue')
        plt.axhline(0.0, color='black', lw=0.8, alpha=0.6)
        plt.scatter([x[0], x[-1]], [0.0, 0.0], color='black', s=20)
        plt.xlabel("x")
        plt.ylabel(r"$\phi(x)$")
        plt.title("Converged Standing Wave (Ground State)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return x, phi_star



def interactive_intensity_plot():
    """Launch interactive widget for intensity/amplitude visualization."""
    widgets.interact(
        intensity_amplitude_plot,
        wavelength=(200, 1000, 10),
        amplitude=(0.5, 3, 0.5)
    )


# =============================================================================
# STANDING WAVE ANIMATION FUNCTIONS
# =============================================================================

def standing_wave(x, t, omega=1, k=1):
    """
    Calculate standing wave amplitude.
    
    Parameters
    ----------
    x : array-like
        Spatial positions
    t : float
        Time
    omega : float
        Angular frequency
    k : float
        Wave number
    
    Returns
    -------
    array
        Standing wave amplitude at each position
    """
    return 2 * np.cos(k * x) * np.cos(omega * t)


def standing_animation(omega=1):
    """
    Create animation of a standing wave.
    
    Parameters
    ----------
    omega : float
        Angular frequency
    
    Returns
    -------
    HTML
        Animation as HTML for display in Jupyter
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    line, = ax.plot(x, standing_wave(x, 0, omega=omega))
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Standing Wave ($\\omega$ = {omega})')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    
    def animate(t):
        line.set_ydata(standing_wave(x, t * 0.1, omega=omega))
        return line,
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# =============================================================================
# RING WAVE ANIMATION FUNCTIONS
# =============================================================================

def ring_wave_animation():
    """
    Create animation of circular ring waves.
    
    Returns
    -------
    HTML
        Animation as HTML for display in Jupyter
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_title('Ring Waves')
    
    circles = []
    
    def animate(frame):
        for c in circles[:]:
            if c.get_radius() > 10:
                c.remove()
                circles.remove(c)
            else:
                c.set_radius(c.get_radius() + 0.2)
                c.set_alpha(max(0, 1 - c.get_radius() / 10))
        
        if frame % 10 == 0:
            circle = plt.Circle((0, 0), 0.1, fill=False, color='blue', alpha=1)
            ax.add_patch(circle)
            circles.append(circle)
        
        return circles
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# =============================================================================
# WAVE INTERFERENCE FUNCTIONS
# =============================================================================

def plot_wave_interference(phases, wavelength, ax):
    """
    Plot interference pattern from multiple waves.
    
    Parameters
    ----------
    phases : array-like
        Phase shifts for each wave
    wavelength : float
        Wavelength in nm
    ax : matplotlib axis
        Axis to plot on
    
    Returns
    -------
    matplotlib axis
        The axis with the plot
    """
    bounds = (-2.5 * wavelength, 2.5 * wavelength)
    wavenumber = 1 / wavelength
    w = 2 * np.pi * wavenumber
    x = np.linspace(*bounds, 200)
    y_1 = 0.5 * np.sin(w * x)
    y_superposition = y_1.copy()
    ax.plot(x, y_1, alpha=0.3)
    
    for phase in list(phases):
        phase_rad = 2 * np.pi * phase
        y_new = 0.5 * np.sin(w * x + phase_rad)
        y_superposition += y_new
        ax.plot(x, y_new, alpha=0.3)
    
    norm = (len(phases) + 1) / 2 if len(phases) > 0 else 1
    y_superposition /= norm
    ax.plot(x, y_superposition, color='blue', linewidth=2)
    ax.set_ylim(-2, 2)
    return ax


def interference_plot_template(theta=0, d=1000, wavelength=400, iterations=2, crystal=False, extra_vars=[]):
    """
    Create a template plot for wave interference visualization.
    
    Parameters
    ----------
    theta : float
        Angle in radians
    d : float
        Spacing in nm
    wavelength : float
        Wavelength in nm
    iterations : int
        Number of wave sources
    crystal : bool
        If True, use crystal (Bragg) geometry
    extra_vars : list
        Extra variables to display
    
    Returns
    -------
    tuple
        (figure, (ax1, ax2, ax_text))
    """
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    ax2 = fig.add_subplot(gs[1, :])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax_text = fig.add_subplot(gs[0, 0])

    if d != 0:
        ax1.set_xlim([-d * 2, d * 2])
        ax1.set_ylim([0, d * 3])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f'Path of parallel rays, $\\theta={np.degrees(theta):.1f}^\\circ$')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')

    if crystal:
        L = 2 * d * np.sin(theta)
    else:
        L = d * np.sin(theta)
    phase = L / wavelength if wavelength != 0 else 0
    
    if iterations > 2:
        orders = np.array([i for i in range(1, iterations)])
        phases = orders * phase
        ax2 = plot_wave_interference(phases, wavelength, ax2)
    else:
        ax2 = plot_wave_interference([phase] if phase != 0 else [], wavelength, ax2)

    ax2.set_xlim(-wavelength * 2.5, wavelength * 2.5)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_title('Interference of parallel rays')
    ax2.set_xlabel('Position (nm)')
    ax2.set_ylabel('Amplitude (not to scale)')

    ax_text.axis("off")
    variable_text = [
        f'$\\lambda={wavelength}$ nm',
        f'$\\theta={np.degrees(theta):.1f}^\\circ$',
    ]
    if crystal:
        variable_text.append(f'$d={d:.3f}$ nm')
    else:
        variable_text.append(f'$d={d:.0f}$ nm')
    if extra_vars:
        variable_text.extend(extra_vars)
    if crystal:
        variable_text.append(f'$L=2d\\sin(\\theta) = {L:.3f}$ nm')
        variable_text.append(f'$2\\theta={2*np.degrees(theta):.1f}^\\circ$')
    else:
        variable_text.append(f'$L=d\\sin(\\theta) = {L:.0f}$ nm')
    ax_text.text(0.5, 0.5, "\n\n".join(variable_text), fontsize=12, ha="center", va="center",
                 transform=ax_text.transAxes)
    plt.tight_layout()

    return fig, (ax1, ax2, ax_text)


def interference_subplot(ax1, theta, d, iterations):
    """
    Add ray paths to interference plot.
    
    Parameters
    ----------
    ax1 : matplotlib axis
        Axis to draw on
    theta : float
        Angle in radians
    d : float
        Spacing
    iterations : int
        Number of rays
    """
    signed_angle = theta
    theta = np.abs(theta)
    height = d * 3
    translations = range(-iterations // 2, iterations // 2 + 1)
    
    for i, translation in enumerate(translations):
        origin = np.array([float(translation) * d, 0.0])
        # Draw ray from each point
        end_point = origin + np.array([height * np.tan(theta), height])
        ax1.plot([origin[0], end_point[0]], [origin[1], end_point[1]], 
                 'b-', alpha=0.6)
        ax1.plot(origin[0], origin[1], 'ko', markersize=4)


def interactive_interference_plot():
    """Launch interactive widget for interference visualization."""
    def plot_interference(theta_deg, d, wavelength):
        theta = np.radians(theta_deg)
        fig, axes = interference_plot_template(theta, d, wavelength, iterations=5)
        interference_subplot(axes[0], theta, d, 5)
        plt.show()
    
    widgets.interact(
        plot_interference,
        theta_deg=(0, 45, 1),
        d=(500, 2000, 100),
        wavelength=(300, 700, 10)
    )


# =============================================================================
# HARMONIC ANALYSIS FUNCTIONS
# =============================================================================

def plot_harmonics(data_dict, title="Harmonic Analysis"):
    """
    Plot harmonic data from CSV files.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping harmonic names to DataFrames
    title : str
        Plot title
    """
    fig, axes = plt.subplots(len(data_dict), 2, figsize=(12, 3 * len(data_dict)))
    
    for i, (name, df) in enumerate(data_dict.items()):
        # Time domain
        axes[i, 0].plot(df.iloc[:, 0], df.iloc[:, 1])
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title(f'{name} - Time Domain')
        
        # Frequency domain
        signal = df.iloc[:, 1].values
        freqs, mags = fourier_transform(signal, sampling_rate=len(signal))
        axes[i, 1].plot(freqs[:len(freqs)//4], mags[:len(mags)//4])
        axes[i, 1].set_xlabel('Frequency')
        axes[i, 1].set_ylabel('Magnitude')
        axes[i, 1].set_title(f'{name} - Frequency Domain')
    
    plt.tight_layout()
    plt.show()
