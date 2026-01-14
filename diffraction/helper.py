import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pandas as pd

def distance_mc_runner(answer_distance, answer_interact, axis_limit=10):
    """
    Validate multiple-choice answers and launch an interactive plot that
    visualizes two points, the segment connecting them, and their Euclidean distance.

    Parameters
    ----------
    answer_distance : {'A','B','C','D'}
        Multiple-choice selection for the distance formula.
        Correct answer is 'C': sqrt((dx)^2 + (dy)^2).

    answer_interact : {'A','B','C','D'}
        Multiple-choice selection for the widgets.interact signature with step=1.
        Correct answer is 'B': v*_ = (-10, 10, 1) for each slider.

    axis_limit : int or float, optional
        Half-range for both axes; plot window is [-axis_limit, axis_limit] in x and y.

    Returns
    -------
    The ipywidgets UI object created by widgets.interact (for notebook environments).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets

    # Map choices to implementations
    distance_forms = {
        'A': lambda v1_x, v1_y, v2_x, v2_y: np.abs(v1_x - v2_x) + np.abs(v1_y - v2_y),           # Manhattan
        'B': lambda v1_x, v1_y, v2_x, v2_y: (v1_x - v2_x)**2 + (v1_y - v2_y)**2,                  # squared distance
        'C': lambda v1_x, v1_y, v2_x, v2_y: np.sqrt((v1_x - v2_x)**2 + (v1_y - v2_y)**2),         # Euclidean (correct)
        'D': lambda v1_x, v1_y, v2_x, v2_y: np.sqrt(v1_x - v2_x) + np.sqrt(v1_y - v2_y),          # invalid generically
    }

    # --- Validate answers ---
    if answer_distance not in distance_forms:
        raise ValueError("ANSWER_DISTANCE must be one of {'A','B','C','D'}.")
    if answer_interact not in {'A','B','C','D'}:
        raise ValueError("ANSWER_INTERACT must be one of {'A','B','C','D'}.")

    if answer_distance != 'C':
        raise AssertionError("Q1 incorrect: The Euclidean distance is option (C).")
    if answer_interact != 'B':
        raise AssertionError("Q2 incorrect: The correct interact signature with step=1 is option (B).")

    # --- Plotting target function ---
    def distance_between_points(v1_x, v1_y, v2_x, v2_y):
        dist = distance_forms[answer_distance](v1_x, v1_y, v2_x, v2_y)

        plt.figure(figsize=(6, 6))
        # Points
        plt.scatter([v1_x], [v1_y], color='tab:blue', label=f"P₁=({v1_x:.1f},{v1_y:.1f})", zorder=3)
        plt.scatter([v2_x], [v2_y], color='tab:orange', label=f"P₂=({v2_x:.1f},{v2_y:.1f})", zorder=3)
        # Connecting segment
        plt.plot([v1_x, v2_x], [v1_y, v2_y], 'k--', lw=1.6, label=f"‖P₁P₂‖ = {dist:.3f}", zorder=2)

        # Axes, grid, labels
        lim = float(axis_limit)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.axhline(0, color='0.75', lw=1)
        plt.axvline(0, color='0.75', lw=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Distance Between Two Points")
        plt.legend(loc='upper left', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.show()

    # --- Build and return the interactive UI ---
    ui = widgets.interact(
        distance_between_points,
        v1_x=(-10, 10, 1),
        v1_y=(-10, 10, 1),
        v2_x=(-10, 10, 1),
        v2_y=(-10, 10, 1),
    )
    return ui

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets

class SuperpositionVisualizer:
    def __init__(self, waves_and_superposition, sampling_rate=64):
        """
        Parameters
        ----------
        waves_and_superposition : callable
            A function with signature
                x, y1, y2, y_superposition = waves_and_superposition(f1, p1, f2, p2)
        sampling_rate : float
            Sampling rate used in the Fourier transform.
        """
        self.waves_and_superposition = waves_and_superposition
        self.sampling_rate = sampling_rate

    def fourier_transform(self, signal):
        """Compute one-sided Fourier magnitude spectrum."""
        T = 1.0 / self.sampling_rate
        n = len(signal)

        fft_signal = np.fft.fft(signal) / n
        frequencies = np.fft.fftfreq(n, T)

        positive_frequencies = frequencies[:n // 2]
        fft_signal_magnitude = np.abs(fft_signal[:n // 2])

        return positive_frequencies, fft_signal_magnitude

    def superposition_plot(self, f1, p1, f2, p2):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        x, y_1, y_2, y_superposition = self.waves_and_superposition(f1, p1, f2, p2)

        # Time-domain plots
        ax1.plot(x, y_1, label='First Wave', alpha=0.6, color='red')
        ax1.plot(x, y_2, label='Second Wave', alpha=0.6, color='orange')
        ax1.plot(x, y_superposition, label='Sum of waves', color='blue')
        ax1.set_title('Wave Superposition')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_ylim(-3, 3)
        ax1.set_xlim(0, 4)
        ax1.legend()

        # Frequency-domain plot
        fourier_x, fourier_y = self.fourier_transform(y_superposition)
        ax2.plot(fourier_x, fourier_y)
        ax2.set_title('Fourier Transform of Resulting Wave')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Contribution')
        ax2.set_xlim(0, 6)
        ax2.set_ylim(0, 1.05)

        plt.subplots_adjust(hspace=0.6)
        plt.show()

    def interactive(self):
        """Interactive widget for varying the parameters."""
        widgets.interact(
            self.superposition_plot,
            f1=(0.0, 5.0, 0.1),
            p1=(0.0, 20.0, 0.1),
            f2=(0.0, 5.0, 0.1),
            p2=(0.0, 20.0, 0.1),
        )

def intensity_amplitude_plot(wavelength,amplitude):
    wavenumber = 1/ wavelength
    w = 2 * np.pi * wavenumber
    x = np.linspace(-1000, 1000, 500)
    y_wave = amplitude * np.sin(w * x)
    y_intensity = y_wave **2
    plt.plot(x,y_wave,label='Wave Amplitude')
    plt.plot(x,y_intensity,label='Wave Intensity')
    plt.axhline(0,color='grey')
    plt.legend()
    plt.ylabel('Intensity and Amplitude (not to scale)')
    plt.xlabel('Position (nm)')
    plt.title('Amplitude and Intensity')
    plt.xlim(-1000,1000)
    plt.ylim(-3,10)
    plt.show()

def interactive_intensity_plot():
    widgets.interact(intensity_amplitude_plot, wavelength=(200,1000,10),amplitude=(0.5,3,0.5))

def plot_wave_interference(phases,wavelength,ax):
    bounds = (-2.5*wavelength,2.5*wavelength)
    wavenumber = 1/wavelength
    w = 2 * np.pi * wavenumber 
    x = np.linspace(*bounds, 200)
    y_1 = 0.5 * np.sin(w * x)
    y_superposition = y_1
    ax.plot(x,y_1,alpha=0.3)
    for i, phase in enumerate(list(phases)):
        phase = 2 * np.pi * phase
        y_new =  0.5 * np.sin(w * x + phase)
        y_superposition += y_new
        plot_version = y_new.copy()
        ax.plot(x,plot_version,alpha=0.3)
    norm = (len(phases) / 2) if len(phases) > 1 else 1 
    y_superposition /= norm
    ax.plot(x, y_superposition)
    ax.set_ylim(-2, 2)
    return ax

def double_slit_plot_template(theta_deg, d=1000, wavelength=400):
    """
    Two-panel figure for the double-slit:
      ax1: geometric path of two rays from two slits
      ax2: time-domain superposition of two waves with phase difference set by θ
    """
    theta_rad = np.radians(theta_deg)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        constrained_layout=True
    )

    # --- Panel 1: Two-slit geometry (reuse interference_subplot with iterations=2)
    interference_subplot(ax1, theta_rad, d, iterations=2)
    ax1.set_title("Double-Slit Geometry")
    ax1.set_xlabel("x (arbitrary units)")
    ax1.set_ylabel("y (arbitrary units)")
    ax1.set_aspect("equal", "box")

    # --- Panel 2: Superposition of two waves -------------------------------
    # Physical phase difference between the two paths:
    # path difference = d sin(θ), so Δφ = 2π (d sinθ) / λ
    delta_phi = 2 * np.pi * d * np.sin(theta_rad) / wavelength

    t = np.linspace(0, 2 * np.pi, 1000)  # "time" axis in arbitrary units
    y1 = np.sin(t)
    y2 = np.sin(t + delta_phi)
    y_sum = y1 + y2

    ax2.plot(t, y1, label="Wave 1", alpha=0.7)
    ax2.plot(t, y2, label="Wave 2", alpha=0.7)
    ax2.plot(t, y_sum, label="Sum", linewidth=2.0)

    ax2.set_xlabel("Time (arb. units)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Superposition of Two Waves at θ = {theta_deg:.1f}°")
    ax2.legend(loc="upper right")

    return fig, (ax1, ax2)


def plot_double_slit(theta):
    d = 1000
    wavelength = 400
    fig, (ax1, ax2) = double_slit_plot_template(
        theta_deg=theta,
        d=d,
        wavelength=wavelength,
    )
    plt.show()

def interactive_double_slit_plot():
    widgets.interact(plot_double_slit, theta=(-90.0, 90.0, 0.1))

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

def interference_plot_template(theta,
                               d,
                               wavelength,
                               crystal: bool = False,
                               extra_vars=None):
    """
    Generic 3-panel template:
      ax1: geometric picture (filled by caller)
      ax2: simple intensity vs θ
      ax_text: explanatory text; optionally crystal-specific with extra variables.

    Parameters
    ----------
    theta : float
        Scattering / incidence angle in radians (used for ax2 curve).
    d : float
        Spacing (slit spacing or plane spacing), in same length units as wavelength.
    wavelength : float
        Wavelength in same units as d.
    crystal : bool, optional
        If True, use Bragg-law language in the text panel.
    extra_vars : list of str, optional
        Extra LaTeX-formatted variable strings to display in the text panel.
    """
    fig, (ax1, ax2, ax_text) = plt.subplots(
        3, 1,
        figsize=(8, 8),
        constrained_layout=True
    )

    # ----------------- ax2: Intensity vs θ -----------------------------------
    thetas_deg = np.linspace(0.1, 90.0, 1000)  # avoid exactly 0 to dodge division by 0
    thetas_rad = np.radians(thetas_deg)

    # Simple two-beam interference intensity pattern as a function of θ
    # path difference = 2 d sin θ  (Bragg-like), phase = 2π * Δ / λ
    delta_phase = 2 * np.pi * (2 * d * np.sin(thetas_rad)) / wavelength
    intensity = (1 + np.cos(delta_phase)) / 2.0  # between 0 and 1

    ax2.plot(thetas_deg, intensity)
    ax2.set_xlabel(r"$\theta$ (degrees)")
    ax2.set_ylabel("Normalized intensity")
    if crystal:
        ax2.set_title("Bragg-like interference intensity")
    else:
        ax2.set_title("Interference intensity vs angle")

    return fig, (ax1, ax2, ax_text)


def interference_subplot(ax1, theta, d, iterations):
    signed_angle = theta          # keep original sign
    theta = np.abs(theta)         # use absolute value for geometry
    
    height = d * 3
    translations = range(-iterations // 2, iterations // 2)

    for i, translation in enumerate(translations):
        origin = np.array([float(translation) * d, 0.0])
        p1 = origin + np.array([-d / 2, 0.0])
        p2 = origin + np.array([d / 2, 0.0])

        p3_x = d * np.cos(theta) * np.cos(theta)
        p3_y = d * np.cos(theta) * np.sin(theta)
        p3 = p1 + np.array([p3_x, p3_y])

        p45_trans = np.array([-np.tan(theta) * height, height])
        p4 = p2 + p45_trans

        if i == 0:
            pairs = [(p2, p4)]
        else:
            pairs = [(p1, p3), (p2, p3), (p3, p4)]

        for p_start, p_end in pairs:
            x = np.array([p_start[0], p_end[0]])
            y = np.array([p_start[1], p_end[1]])

            if signed_angle > 0:
                ax1.plot(-x, y)
            else:
                ax1.plot(x, y)

    ax1.set_aspect("equal", "box")
    return ax1

def diffraction_plot_template(theta_deg, d=1000, wavelength=400, iterations=25, N=25):
    """
    Three-panel figure:
      ax1: geometric interference diagram (many slits)
      ax2: N-slit diffraction intensity vs angle
      ax3: time-domain superposition of N waves at the chosen θ
    """
    theta_rad = np.radians(theta_deg)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        constrained_layout=True
    )

    # --- Panel 1: Geometry / rays -------------------------------------------
    interference_subplot(ax1, theta_rad, d, iterations)
    ax1.set_title(f"Interference from {iterations} slits")
    ax1.set_xlabel("x (arbitrary units)")
    ax1.set_ylabel("y (arbitrary units)")
    ax1.set_aspect("equal", "box")

    # --- Panel 2: N-slit diffraction intensity vs θ -------------------------
    thetas = np.linspace(-70, 70, 2000)    # degrees
    th_rad = np.radians(thetas)

    beta = np.pi * d * np.sin(th_rad) / wavelength
    beta_safe = np.where(np.isclose(beta, 0.0), 1e-12, beta)

    intensity = (np.sin(N * beta_safe) / np.sin(beta_safe)) ** 2
    intensity /= intensity.max()

    ax2.plot(thetas, intensity)
    ax2.set_xlabel(r"$\theta$ (degrees)")
    ax2.set_ylabel("Normalized intensity")
    ax2.set_title(f"Diffraction pattern, N = {N}")

    # --- Panel 3: Superposition of N waves at this θ ------------------------
    # Phase increment corresponding to this particular viewing angle θ
    beta_theta = np.pi * d * np.sin(theta_rad) / wavelength

    # Time axis for visualization (arbitrary units, one "period-scale" window)
    t = np.linspace(0, 2 * np.pi, 1000)

    # Build N waves with phase offset n * beta_theta
    waves = []
    for n in range(N):
        waves.append(np.sin(t + n * beta_theta))
    waves = np.array(waves)

    # Sum and normalize
    superposed = waves.sum(axis=0) / N

    # Plot individual waves faintly, superposition in bold
    for n in range(N):
        ax3.plot(t, waves[n], alpha=0.2, linewidth=0.8)
    ax3.plot(t, superposed, linewidth=2.0)

    ax3.set_xlabel("Time (arb. units)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title(f"Superposition of {N} Waves at θ = {theta_deg:.1f}°")

    return fig, (ax1, ax2, ax3)

def plot_diffraction(theta):
    d = 1000
    wavelength = 400
    fig, (ax1, ax2, ax3) = diffraction_plot_template(
        theta_deg=theta,
        d=d,
        wavelength=wavelength,
        iterations=25,
        N=25,
    )
    plt.show()

def interactive_diffraction_plot():
    widgets.interact(plot_diffraction, theta=(-70.0, 70.0, 0.1))

from matplotlib.patches import Arc
def bragg_intensity_subplot(ax, d, wavelength):
    """
    Plot a simple two-beam Bragg-like interference intensity:
        Δℓ(θ) = 2 d sin θ
        φ(θ) = 2π Δℓ / λ
        I(θ) ∝ (1 + cos φ) / 2
    """
    thetas_deg = np.linspace(0.1, 90.0, 1000)   # avoid exactly 0 to dodge division by zero
    thetas_rad = np.radians(thetas_deg)

    path_diff = 2.0 * d * np.sin(thetas_rad)          # Δℓ(θ)
    phase = 2.0 * np.pi * path_diff / wavelength      # φ(θ)
    intensity = 0.5 * (1.0 + np.cos(phase))           # normalized between 0 and 1

    ax.plot(thetas_deg, intensity)
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(r"Bragg-like intensity: $2d\sin\theta = n\lambda$")
    ax.set_ylim(-0.05, 1.05)

    return ax

def two_d_crystal_lattice(a,b,gamma,title='2D Crystal Lattice'):
    fig,ax = plt.subplots(figsize=(3,3))
    gamma=np.radians(gamma)
    tv_b = np.array([b,0])
    
    x_a = a * np.cos(gamma)
    y_a = a * np.sin(gamma)
    tv_a = np.array([x_a,y_a])

    origin = np.array([0.0,0.0])

    tv_ab = tv_a + tv_b
    unitcell_points = [tv_a,tv_ab,tv_b]
    unit_x = [point[0] for point in unitcell_points]
    unit_y = [point[1] for point in unitcell_points]
    ax.plot(unit_x,unit_y,color='red')
    ax.quiver(0, 0, tv_a[0], tv_a[1], color='b', angles='xy', 
              scale_units='xy', scale=1, width=0.015, label=f"$a={a:.2f}"+r"\AA$")
    ax.quiver(0, 0, tv_b[0], tv_b[1], color='r', angles='xy', 
              scale_units='xy', scale=1, width=0.015, label=f"$b={b:.2f}"+r"\AA$")
    arc = Arc((0, 0), width=0.5*a, height=0.5*a, angle=0, theta1=0, 
              theta2=np.degrees(gamma), color='black', linewidth=1.5,label=f'$\\gamma={np.degrees(gamma):.1f}\\degree$')
    ax.add_patch(arc)
    
    translations = 10
    points = []
    for i in range(-translations,translations+1):
        for j in range(-translations,translations+1):
            a = i * tv_a
            b = j * tv_b 
            point = a + b
            points.append(point)
        
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    ax.grid()
    ax.scatter(x,y)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_title(title)
    ax.set_xlabel('x (Angstrom)')
    ax.set_ylabel('y (Angstrom)')
    ax.set_aspect('equal')

    ax.legend(loc="lower right", bbox_to_anchor=(1, 0),markerscale=0.5)
    plt.show()

def interactive_crystal_plot():
    widgets.interact(two_d_crystal_lattice,
                     a=(1,2,0.1),
                     b=(1,2.0,0.1),
                    gamma=(30,150,1)
                    )

def point_reflection_subplot(ax1,x1,y1,x2,y2,theta):
    x = [x1,x2]
    y = [y1,y2]
    ax1.scatter(x,y)
    
    right_trans = 50 * np.array([np.cos(theta),np.sin(theta)])
    left_trans = 50 * np.array([-np.cos(theta),np.sin(theta)])
    p1 = np.array([x1,y1])
    p2 = np.array([x2,y2])
    points = [p1,p2]
    colors = ['blue','orange']
    for i, p in enumerate(points):
        p_right = p + right_trans
        p_left = p + left_trans
        light_path = [p_right,p,p_left]
        x_light = [xy[0] for xy in light_path]
        y_light = [xy[1] for xy in light_path]
        ax1.plot(x_light,y_light,color =colors[i])
    return ax1

def reflection_two_points(x2=0.0, y2=0.0, theta=45):
    # Fixed first point at origin
    x1 = 0.0
    y1 = 0.0

    # Degrees → radians
    theta_rad = np.radians(theta)

    # Effective spacing along normal: use vertical separation as d
    d = np.abs(y1 - y2)      # same units as wavelength (here: nm)
    wavelength = 0.154       # nm (1.54 Å)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(7, 9),
        constrained_layout=True,
    )

    # ---- Panel 1: geometry of two reflection points ----
    point_reflection_subplot(ax1, x1, y1, x2, y2, theta_rad)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect("equal", "box")
    ax1.set_title("Reflection from Two Points")
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')

    # annotate dx, dy
    dy = np.abs(y1 - y2)
    dx = np.abs(x1 - x2)
    ax1.text(
        0.02, 0.98,
        rf"$d_y = {dy:.3f}\,\mathrm{{nm}}$" + "\n" +
        rf"$d_x = {dx:.3f}\,\mathrm{{nm}}$",
        transform=ax1.transAxes,
        va="top",
        ha="left",
    )

    # ---- Panel 2: Bragg intensity vs θ for this d ----
    bragg_intensity_subplot(ax2, d=d, wavelength=wavelength)

    # ---- Panel 3: Interference for this particular θ and d ----
    two_wave_interference_subplot(ax3, d=d, wavelength=wavelength, theta_rad=theta_rad)

    plt.show()
    return fig, (ax1, ax2, ax3)

def interactive_reflection_plot():
    widgets.interact(
        reflection_two_points,
        x2=(-1.0, 1.0, 0.01),
        y2=(-1.0, 1.0, 0.01),
        theta=(1.0, 90.0, 0.1),
    )

def self_similar_bragg(origin,D,theta,ax):
    p_above = origin + np.array([0,D])
    x_right = D * np.sin(theta) * np.cos(theta)
    y_right = D * np.sin(theta) ** 2
    p_right = np.array([x_right,y_right])
    p_right_long = 1000 * p_right
    p_left = np.array([-x_right,y_right])
    p_left_long = 1000 * p_left
    
    p_right += origin
    p_right_long += origin
    p_left += origin
    p_left_long += origin
    
    points = [
        p_above, p_right, p_right_long, p_left, p_left_long,
    ]
    pairs = [
        (origin,p_above,'black'),(origin,p_left,'orange'),(origin,p_right,'orange'),
        (p_above,p_left,'green'),(p_above,p_right,'green'),(p_left,p_left_long,'blue'),
        (p_right,p_right_long,'blue'),
    ]    
    for pair in pairs:
        x = np.array([point[0] for point in pair if type(point) is np.ndarray])
        y = np.array([point[1] for point in pair if type(point) is np.ndarray])
        color = pair[2]  
        ax.plot(x,y,color=color)
        
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    
    ax.set_aspect('equal', adjustable='box')
    distance = 2 * D * np.sin(theta)
    return ax, distance

from matplotlib.ticker import MultipleLocator

from matplotlib.ticker import MultipleLocator

def bragg_plot(theta):
    # theta comes in degrees from the widget
    theta_rad = np.radians(theta)

    # Interplanar spacing (same units as wavelength)
    d = 0.2
    wavelength = 0.154

    # 3 panels: geometry, Bragg intensity vs θ, interference at this θ
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(7, 9),
        constrained_layout=True,
    )

    # ---- Panel 1: self-similar Bragg construction (geometry) ----
    mirror_normal_distance = d

    for i in range(-2, 3):
        for j in range(-4, 3):
            x_trans = 0.3 * i
            y_trans = mirror_normal_distance * j
            origin = np.array([x_trans, y_trans])
            ax1, distance = self_similar_bragg(origin, d, theta_rad, ax1)

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.xaxis.set_major_locator(MultipleLocator(0.3))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.set_aspect("equal", "box")
    ax1.set_title("Self-similar Bragg Construction")

    # ---- Panel 2: Bragg intensity vs θ for this spacing d ----
    bragg_intensity_subplot(ax2, d=d, wavelength=wavelength)

    # ---- Panel 3: Two-wave interference for this θ and d ----
    two_wave_interference_subplot(ax3, d=d, wavelength=wavelength, theta_rad=theta_rad)

    plt.show()
    return fig, (ax1, ax2, ax3)

def interactive_bragg_plot():
    widgets.interact(bragg_plot, theta=(1.0, 70.0, 0.1))


def two_wave_interference_subplot(ax, d, wavelength, theta_rad):
    """
    For a fixed spacing d and angle theta_rad, plot:
        y1(t) = sin(t)
        y2(t) = sin(t + Δφ)
        y_sum = y1 + y2
    where Δφ = 2π * (2 d sin θ) / λ.
    """
    path_diff = 2.0 * d * np.sin(theta_rad)          # Δℓ
    delta_phi = 2.0 * np.pi * path_diff / wavelength # Δφ

    t = np.linspace(0, 2 * np.pi, 1000)
    y1 = np.sin(t)
    y2 = np.sin(t + delta_phi)
    y_sum = y1 + y2

    ax.plot(t, y1, label="Wave 1", alpha=0.7)
    ax.plot(t, y2, label="Wave 2", alpha=0.7)
    ax.plot(t, y_sum, label="Sum", linewidth=2.0)

    ax.set_xlabel("Time (arb. units)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        r"Two-wave interference for "
        rf"$d = {d:.3f}$, $\theta = {np.degrees(theta_rad):.1f}^\circ$"
    )
    ax.legend(loc="upper right")

    return ax

def plot_lattice(fig,ax,a=4.09):    
    colors = ['red','green','blue']
    for i in range(3):
        points = []
        for j in range(3):
            for k in range(3):
                points.append(a*np.array([j,i,k]))
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        z = [point[2] for point in points]
        
        ax.scatter(x,y,z,c=y,cmap='magma',vmin=0.0, vmax=3.0*a,s=100)

    x_c = 0.5
    y_c = 0.5
    z_c = 0.5
    points = []
    translations = [-0.5,0,0.5]
    for i in translations:
        for j in translations:
            for k in translations:
                if sum(np.abs(np.array([i,j,k]))) == 0.5:
                    points.append(np.array([x_c+i,y_c+j,z_c+k]))

    new_points = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for point in points:
                    translation = np.array([i,j,k])
                    new_point = translation + np.array(point)
                    new_points.append(a*new_point)
    
    x = [point[0] for point in new_points]
    y = [point[1] for point in new_points]
    z = [point[2] for point in new_points]
        
    ax.scatter(x,y,z,c=y,cmap='magma',vmin=0.0, vmax=3.0*a,s=100)
                

    unitcell_edges = [
        ([0,0,0],[a,0,0]),
        ([0,0,0],[0,a,0]),
        ([0,0,0],[0,0,a]),
        ([a,0,0],[a,a,0]),
        ([a,0,0],[a,0,a]),
        ([0,a,0],[a,a,0]),
        ([0,a,0],[0,a,a]),
        ([0,0,a],[a,0,a]),
        ([0,0,a],[0,a,a]),
        ([a,a,0],[a,a,a]),
        ([0,a,a],[a,a,a]),
        ([a,0,a],[a,a,a]),
    ]
    unitcell_edges = [(np.array(edge[0]),np.array(edge[1])) for edge in unitcell_edges]
    for edge in unitcell_edges:
        x = [point[0] for point in edge]
        y = [point[1] for point in edge]
        z = [point[2] for point in edge]
        ax.plot(x,y,z,color='black',alpha=0.5)
    
    outer_edges = [(2*edge[0],2*edge[1]) for edge in unitcell_edges]
    for edge in outer_edges:
        x = [point[0] for point in edge]
        y = [point[1] for point in edge]
        z = [point[2] for point in edge]
        ax.plot(x,y,z,color='black',alpha=0.5)

    return fig, ax

def truncate_points(xx,yy,zz,a=4.09):
    bounds = (a,a,a)
    outer_xx = []
    outer_yy = []
    outer_zz = []
    for i in range(xx.shape[0]):
        inner_xx = []
        inner_yy = []
        inner_zz = []
        for j in range(yy.shape[1]):
            x = xx[i,j]
            y = yy[i,j]
            z = zz[i,j]
            cond1 = x >= 0 and x <= bounds[0]
            cond2 = y >= 0 and y <= bounds[1]
            cond3 = z >= 0 and z <= bounds[2]
            if cond1 and cond2 and cond3:
                point = np.array([x,y,z])
                inner_xx.append(point[0])
                inner_yy.append(point[1])
                inner_zz.append(point[2])
            else:
                print(f'point {x,y,z} is out of bounds!')
        outer_xx.append(inner_xx)
        outer_yy.append(inner_yy)
        outer_zz.append(inner_zz)
        
    new_xx = np.array(outer_xx)
    new_yy = np.array(outer_yy)
    new_zz = np.array(outer_zz)
    return new_xx, new_yy, new_zz

def plot_planes(fig,ax,hkl,a=4.09):
    h, k, l = hkl
    wavelength = 1.54

    if h != 0:
        norm_x = 1 / h
    else:
        norm_x = 0
    if k != 0:
        norm_y = 1/k
    else:
        norm_y =  0
    if l != 0:
        norm_z = 1/l
    else:
        norm_z = 0

    

    if not (h == 0 and k == 0 and l == 0):
        mirror_vector = a * np.array([norm_x, norm_y, norm_z])
        
        d = a/np.sqrt(h**2 + k**2 + l**2)
        normal_vector = mirror_vector / np.linalg.norm(mirror_vector)
        translation_iterations = max((np.abs(h)+np.abs(k)+np.abs(l)) * 2,3) + 2
        colormap = plt.get_cmap('viridis', translation_iterations)

    else:
        d = np.inf
    
    if l != 0:
        theta_d_z = np.arccos(
            np.linalg.norm(
                np.dot(normal_vector,np.array([0,0,1]))
            )
        )
        zz_translation = a / l  
        
        xx, yy = np.meshgrid(np.linspace(0, 2*a, 11), np.linspace(0, 2*a, 11))
        for i in range(-translation_iterations,translation_iterations):
            zz = -(h * xx + k * yy) / l + zz_translation * i 
            
            mask = (zz >= -0.01) & (zz <= 2.01*a) & (xx >= -0.01) & (xx <= 2.01*a) & (yy >= -0.01) & (yy <= 2.01*a)
            zz = np.where(mask,zz,np.nan)
            
            ax.plot_surface(xx, yy, zz, alpha=0.5, color=colormap(np.abs(i)))
            ax.plot_wireframe(xx, yy, zz, alpha=0.5, color=colormap(np.abs(i)))

    
    elif k != 0:
        theta_d_y = np.arccos(
            np.linalg.norm(
                np.dot(normal_vector,np.array([0,1,0]))
            )
        )
        yy_translation = a / k
        xx, zz = np.meshgrid(np.linspace(0, 2*a, 10), np.linspace(0, 2*a, 11))
        for i in range(-translation_iterations,translation_iterations):
            yy = -(h * xx) / k + yy_translation * i 
            mask = (zz >= -0.01) & (zz <= 2.01*a) & (xx >= -0.01) & (xx <= 2.01*a) & (yy >= -0.01) & (yy <= 2.01*a)
            yy = np.where(mask,yy,np.nan)
            ax.plot_surface(xx, yy, zz, alpha=0.5, color=colormap(np.abs(i)))
            ax.plot_wireframe(xx, yy, zz, alpha=0.5, color=colormap(np.abs(i)))

    elif h != 0:
        xx_translation = a/h
        
        yy, zz = np.meshgrid(np.linspace(0, 2*a, 11), np.linspace(0, 2*a, 11))
        
        for i in range(-translation_iterations,translation_iterations):
            xx = xx_translation * i 
            mask = (zz >= -0.01) & (zz <= 2.01*a) & (xx >= -0.01) & (xx <= 2.01*a) & (yy >= -0.01) & (yy <= 2.01*a)
            xx = np.where(mask,xx,np.nan)
            ax.plot_surface(xx, yy, zz, alpha=0.5, color=colormap(np.abs(i)))
            ax.plot_wireframe(xx, yy, zz, alpha=0.5, color=colormap(np.abs(i)))
    else:
        pass
    h_text = f'{h}' if h >= 0 else r'$\overline{'+f'{abs(h)}'+r'}$'
    k_text = f'{k}' if k >= 0 else r'$\overline{'+f'{abs(k)}'+r'}$'
    l_text = f'{l}' if l >= 0 else r'$\overline{'+f'{abs(l)}'+r'}$'
    ax.set_title(F'Mirror Planes | hkl=({h_text}{k_text}{l_text})')
    ax.set_xlim(0,2*a)
    ax.set_ylim(0,2*a)
    ax.set_zlim(0,2*a)
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('x (Angstrom)')
    ax.set_ylabel('y (Angstrom)')
    ax.set_zlabel('z (Angstrom)')
    
    theta_1 = np.degrees(np.arcsin(wavelength/(2*d))) 
    line_1 = f'$\\lambda = {wavelength} \\AA$'
    line_2 = r'$d = \frac{a}{\sqrt{h^2+k^2+l^2}} = '+f'{d:.2f}$'
    line_3 = r'$\theta_1 = sin^{-1}(\frac{\lambda}{2d}) = '+f'{theta_1:.2f}\\degree$'
    line_4 = r'$2\theta = ' + f'{theta_1 * 2:.2f}\\degree$'
    fig.text(0.1,0,'\n'.join([line_1,line_2]))
    fig.text(0.6,0,'\n'.join([line_3,line_4]))

    return fig, ax 

def visualize_mirror_planes(h,k,l):
    plt.cla()
    fig = plt.figure(figsize=(4,5))
    a=4.0
    ax = fig.add_subplot(projection='3d')
    plot_lattice(fig,ax,a)
    hkl = (h,k,l)
    plot_planes(fig,ax,hkl,a)
    plt.show()
    
def interactive_mirror_plot():
    widgets.interact(visualize_mirror_planes,h=(-2,2,1),k=(-2,2,1),l=(-2,2,1))

def pick_peaks(x,y):
    peaks = []
    intensities = []
    thresh = 3
    is_peak = False
    max_intens = 0
    max_index = 0
    for i, intensity in enumerate(y):
        if not is_peak:
            if intensity > thresh:
                is_peak = True
                max_intens = intensity
                max_index = i
        if is_peak:
            if intensity < thresh:
                is_peak = False
                peaks.append(x[max_index])
                intensities.append(max_intens)
                max_intens = 0
            elif intensity > max_intens:
                max_intens = intensity
                max_index = i
    return pd.DataFrame({
        '2theta': peaks,
        'intensity' : intensities,
    })

def calculate_distances(two_theta,wavelength):
    two_theta = np.radians(two_theta)
    theta = two_theta / 2
    d = wavelength / (2 * np.sin(theta))
    return d

def generate_r_series():
    hkl_vals = [
        (1,0,0),(1,1,0),(1,1,1),(2,0,0),
        (2,1,0),(2,1,1),(2,2,0),(2,2,1),
        (2,2,2),(3,0,0),(3,1,0),(3,1,1),
        (3,2,0),(3,2,1),(3,2,2),(3,3,1),
    ]
    r_with_hkl = []
    for hkl in hkl_vals:
        h,k,l = hkl
        r = 1 / np.sqrt(h**2+k**2+l**2)
        r_with_hkl.append((r,hkl))
    r_with_hkl.sort(key=lambda x: x[0],reverse=True)
    r_series = np.array([r_hkl[0] for r_hkl in r_with_hkl])
    hkl_series = [r_hkl[1] for r_hkl in r_with_hkl]
    return r_series, hkl_series

def test_close(n1,n2,thresh=0.03):
    diff = np.abs(n1 - n2)
    if diff < thresh:
        return True
    return False

def index_hkl(d_points,starting_index=0):
    if starting_index > 10:
        raise ValueError('Could not converge on indices')
    r_s,hkl_s = generate_r_series()
    prop_factor = d_points[0] / r_s[starting_index]
    r_s = r_s[starting_index:]
    hkl_s = hkl_s[starting_index:]
    r_prop = r_s * prop_factor 
    a = prop_factor
    labeled_d_points = []
    
    for d in d_points:
        for i, r in enumerate(r_prop):
            if test_close(r,d):
                labeled_d_points.append((d,hkl_s[i]))
                break
    d_vals = [point_label[0] for point_label in labeled_d_points]
    list_hkl_tuples = [point_label[1] for point_label in labeled_d_points]
    list_hkl = [int(f'{hkl[0]}{hkl[1]}{hkl[2]}') for hkl in list_hkl_tuples]
    #recursion!
    if len(labeled_d_points) != len(d_points):
        list_hkl, a = index_hkl(d_points,starting_index+1)
        #it's like a dream inside a dream inside a dream inside a ....

    return list_hkl, a

def classify_cubic_spectra(hkl_list):
    could_be_bcc = True
    could_be_fcc = True
    for hkl in hkl_list:
        if type(hkl) is int or type(hkl) is str:
            hkl = [int(str(hkl)[i]) for i in range(3)]
        hkl = np.array(hkl)
        hkl_parity = hkl % 2
        all_same_parity = False
        if np.array_equal(hkl_parity,np.array([0,0,0])) or np.array_equal(hkl_parity,np.array([1,1,1])):
            all_same_parity = True
        if not all_same_parity:
            could_be_fcc = False
            
        sum_hkl = hkl.sum()
        sum_hkl_parity = sum_hkl % 2
        if not sum_hkl_parity == 0:
            could_be_bcc = False

    if could_be_bcc and could_be_fcc:
        raise ValueError("oh no")
    elif could_be_bcc:
        result = 'body centered cubic'
    elif could_be_fcc:
        result = 'face centered cubic'
    else:
        result = 'simple cubic'
    return result

def plot_spectrum(x, y):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y)
    ax.set_title(r'intensity vs $2\theta$')
    ax.set_xlabel(r'$2\theta$')
    ax.set_ylabel('intensity')
    plt.show()
    return fig, ax
