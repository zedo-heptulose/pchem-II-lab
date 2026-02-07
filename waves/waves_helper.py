import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from matplotlib import animation
from IPython.display import HTML

# Superposition Plot
def superposition_plot(f1, p1, f2, p2):
    fig, (ax1, ax2) = plt.subplots(2,1)
    x, y_1, y_2, y_superposition = waves_and_superposition(f1,p1,f2,p2)

    ax1.plot(x,y_1,label='First Wave',alpha=0.6,color='red')
    ax1.plot(x,y_2,label='Second Wave',alpha=0.6,color='orange')
    ax1.plot(x, y_superposition,label='Sum of waves',color='blue')
    ax1.set_title('Wave Superposition')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-3, 3)
    ax1.set_xlim(0,4)

    fourier_x, fourier_y = fourier_transform(y_superposition)
    ax2.plot(fourier_x,fourier_y)
    ax2.set_title('Fourier Transform of Resulting Wave')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Contribution')
    ax2.set_xlim(0,6)
    ax2.set_ylim(0,1.05)

    plt.subplots_adjust(hspace=0.6)
    plt.show()

def waves_and_superposition(f1,p1,f2,p2):
    #creating list of x values for use with our functions
    x = np.linspace(0, 10, 640) 
    #transforming to angular frequencies for use in sine
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
   
    y_1 = np.sin(w1 * x + p1)
    y_2 = np.sin(w2 * x + p2)
    y_superposition = y_1 + y_2
    
    return x, y_1, y_2, y_superposition

def interactive_superposition_plot():
    return widgets.interact(superposition_plot, f1=(0,5,0.1), p1=(0,20,0.1),f2=(0,5,0.1),p2=(0,20,0.1))

def fourier_transform(signal):
    sampling_rate = 64  
    T = 1 / sampling_rate  
    
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_signal = fft_signal / n 
    
    frequencies = np.fft.fftfreq(n, T)
    
    positive_frequencies = frequencies[:n // 2]
    fft_signal_magnitude = np.abs(fft_signal[:n // 2])
    return positive_frequencies, fft_signal_magnitude

def standing_animation(wavelength=0.5, Nframes=90, interval_ms=50):
    """
    Generate and display an animation of a standing wave for a given wavelength.

    Parameters
    ----------
    wavelength : float
        Wavelength of the wave. Smaller wavelengths mean more oscillations
        in the viewing window. Default is 0.5.

    Nframes : int, optional
        Number of frames in the animation. A higher number results
        in smoother motion but requires more computation. Default is 90.

    interval_ms : int, optional
        Time interval between animation frames in milliseconds.
        Controls the playback speed. Default is 50.
    """
    # -------- wave parameters --------
    # Using c = 1 units, so frequency nu = 1/wavelength
    nu = 1.0 / wavelength
    x = np.linspace(-1, 1, 200)  # fixed domain; shows more wavelengths as λ decreases
    # ---------------------------------

    # Traveling waves (sine => node at x=0 in the sum)
    # E(x,t) = sin(2π(x/λ - νt)) for right-moving wave
    E_r = lambda x, t: np.sin(2 * np.pi * (x / wavelength - nu * t))   # right-moving
    E_l = lambda x, t: np.sin(2 * np.pi * (x / wavelength + nu * t))   # left-moving
    E_sum = lambda x, t: E_r(x, t) + E_l(x, t)   # standing wave
    
    # Build figure/axes
    fig, ax = plt.subplots(figsize=(9, 3.75), dpi=72)
    (line_r,)   = ax.plot([], [], label="Right-moving")
    (line_l,)   = ax.plot([], [], label="Left-moving")
    (line_sum,) = ax.plot([], [], lw=2, label="Standing wave (sum)")
    
    # markers at x=0 for each curve (1-element sequences in updates)
    (point_r,)   = ax.plot([], [], "o", ms=5, color=line_r.get_color())
    (point_l,)   = ax.plot([], [], "o", ms=5, color=line_l.get_color())
    (point_sum,) = ax.plot([], [], "o", ms=5, color=line_sum.get_color())
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-2.2, 2.2)
    ax.set_xticks([])                 # remove x ticks
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$E(x,t)$")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Drive animation by phase so the loop closes exactly
    phases = np.linspace(0.0, wavelength, Nframes, endpoint=False)  # one full period (T = λ/c = λ)
    
    def init():
        # start with empty lines/points
        for ln in (line_r, line_l, line_sum):
            ln.set_data([], [])
        for pt in (point_r, point_l, point_sum):
            pt.set_data([], [])
        return line_r, line_l, line_sum, point_r, point_l, point_sum
    
    def animate(t):
        Er = E_r(x, t)
        El = E_l(x, t)
        Es = Er + El
    
        line_r.set_data(x, Er)
        line_l.set_data(x, El)
        line_sum.set_data(x, Es)
    
        # markers at x=0 (use 1-element sequences)
        point_r.set_data([0.0], [E_r(0.0, t)])
        point_l.set_data([0.0], [E_l(0.0, t)])
        point_sum.set_data([0.0], [E_sum(0.0, t)])  # always ~ 0 because sin(0)=0
    
        return line_r, line_l, line_sum, point_r, point_l, point_sum
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=phases, interval=interval_ms, blit=False
    )
    
    # Render as HTML and suppress extra static figure
    html = ani.to_jshtml()
    plt.close(fig)
    return HTML(html)

def ring_wave_animation(R=1.0, a=0.18, m=4, c=1.0, Nframes=40, Nt=120):
    """
    Generate an animation of a standing or traveling wave wrapped around a circular ring.

    Parameters
    ----------
    R : float, optional
        Base radius of the ring (mean distance from center). Default is 1.0.

    a : float, optional
        Radial modulation amplitude — how far the wave oscillates inward/outward
        relative to the base radius. Default is 0.18.

    m : int, optional
        Mode number (number of wavelengths around the loop). Determines the
        number of nodes along the ring. Default is 4.

    c : float, optional
        Wave propagation speed. Default is 1.0.

    Nframes : int, optional
        Number of frames in the animation (higher values yield smoother motion
        but increase rendering time and file size). Default is 40.

    Nt : int, optional
        Number of angular sampling points around the ring (higher values yield
        smoother curves but heavier computation). Default is 120.
    """
    omega = c*m
    theta = np.linspace(0, 2*np.pi, Nt, endpoint=False)
    
    # traveling and standing fields on the ring
    E_right = lambda th, t: np.cos(m*th - omega*t)      # clockwise
    E_left  = lambda th, t: np.cos(m*th + omega*t)      # counter-clockwise
    E_sum   = lambda th, t: E_right(th,t) + E_left(th,t)# standing = 2 cos(mθ) cos(ωt)
    
    cos_mtheta = np.cos(m*theta)
    node_idx = np.where(np.sign(cos_mtheta[:-1]) * np.sign(cos_mtheta[1:]) < 0)[0]
    node_angles = theta[node_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), dpi=80, subplot_kw={"aspect":"equal"})
    titles = ["Right-moving", "Left-moving", "Standing (sum)"]
    lines, markers = [], []
    
    
    for ax, ttl in zip(axes, titles):
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl)
        
        (ln,) = ax.plot([], [], lw=2)
        lines.append(ln)
        if ttl.startswith("Standing"):
            mk = []
            for ang in node_angles:
                mk.append(ax.plot([R*np.cos(ang)], [R*np.sin(ang)], "k.", ms=5, alpha=0.9)[0])
            markers.append(mk)
        else:
            markers.append([])
    
    phases = np.linspace(0, 2*np.pi, Nframes, endpoint=False)
    
    def ring_xy(field_vals):
        r = R + a*field_vals
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        # append first point to close the stroke
        return np.append(x, x[0]), np.append(y, y[0])
    
    def init():
        for ln in lines: ln.set_data([], [])
        return lines + sum(markers, [])
    
    def animate(phi):
        t = phi/omega
        xr, yr = ring_xy(E_right(theta, t))
        xl, yl = ring_xy(E_left(theta, t))
        xs, ys = ring_xy(E_sum(theta, t))
        lines[0].set_data(xr, yr)
        lines[1].set_data(xl, yl)
        lines[2].set_data(xs, ys)
        return lines + sum(markers, [])
    
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=phases, interval=50, blit=False)
    
    html = ani.to_jshtml()
    plt.close(fig)
    return HTML(html)

def spherical_harmonic_plot(l, m):
    """
    Plot a 3D surface visualization of the real spherical harmonic Y_l^m.

    The plot shows the angular part of atomic orbitals, with the radial distance
    from the origin representing |Y_l^m| and colors indicating the sign
    (positive = blue, negative = orange).

    Parameters
    ----------
    l : int
        Degree of the harmonic (l >= 0). This corresponds to the angular
        momentum quantum number in atomic orbitals:
        l=0 -> s orbitals, l=1 -> p orbitals, l=2 -> d orbitals, etc.

    m : int
        Order of the harmonic (-l <= m <= l). This corresponds to the
        magnetic quantum number in atomic orbitals.

    Returns
    -------
    None
        Displays a 3D matplotlib figure.
    """
    from scipy.special import sph_harm
    
    # Validate inputs
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    
    # Create grid of angles
    # theta = azimuthal angle [0, 2pi], phi = polar angle [0, pi]
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Compute spherical harmonic (scipy uses physics convention)
    Y = sph_harm(m, l, theta_grid, phi_grid)
    
    # Take real part (for m != 0, this gives the "real" spherical harmonics
    # that correspond to px, py, dxy, etc.)
    if m > 0:
        Y_real = np.real(Y) * np.sqrt(2) * (-1)**m
    elif m < 0:
        Y_real = np.imag(Y) * np.sqrt(2) * (-1)**m
    else:
        Y_real = np.real(Y)
    
    # Use |Y| as radial distance, sign for coloring
    r = np.abs(Y_real)
    
    # Convert to Cartesian coordinates
    x = r * np.sin(phi_grid) * np.cos(theta_grid)
    y = r * np.sin(phi_grid) * np.sin(theta_grid)
    z = r * np.cos(phi_grid)
    
    # Create color array based on sign
    colors = np.where(Y_real >= 0, 1.0, 0.0)
    
    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a diverging colormap centered at 0.5
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#ff7f0e', '#1f77b4'])  # orange for negative, blue for positive
    
    ax.plot_surface(x, y, z, facecolors=cmap(colors), 
                    rstride=1, cstride=1, alpha=0.8, linewidth=0)
    
    # Set equal aspect ratio
    max_range = np.max(r) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Spherical Harmonic $Y_{l}^{m}$ (l={l}, m={m})')
    
    plt.tight_layout()
    plt.show()


def photon_energy(L, n):
    """
    Calculate the photon energy for mode n in a cavity of length L.
    
    For a cavity with mirrors at both ends, only wavelengths λ_n = 2L/n are allowed.
    This function computes the corresponding photon energy E = hc/λ and identifies
    the region of the electromagnetic spectrum.
    
    Parameters
    ----------
    L : float
        Length of the cavity in nanometers.
    n : int
        Mode number (n = 1, 2, 3, ...).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'wavelength_nm': float, wavelength in nanometers
        - 'energy_eV': float, energy in electron volts
        - 'spectrum': str, region of electromagnetic spectrum
    
    Examples
    --------
    >>> photon_energy(500, 1)  # 500 nm cavity, mode 1
    """
    # hc in convenient units: eV·nm
    hc = 1240  # eV·nm
    
    # Validate input
    if n < 1:
        raise ValueError(f"Mode number n must be a positive integer, got {n}")
    if L <= 0:
        raise ValueError(f"Cavity length L must be positive, got {L}")
    
    # Calculate wavelength (nm) and energy (eV)
    wavelength_nm = 2 * L / n
    energy_eV = hc / wavelength_nm
    
    # Determine spectrum region
    if wavelength_nm < 10:
        spectrum = "X-ray"
    elif wavelength_nm < 400:
        spectrum = "Ultraviolet (UV)"
    elif wavelength_nm < 450:
        spectrum = "Violet"
    elif wavelength_nm < 495:
        spectrum = "Blue"
    elif wavelength_nm < 570:
        spectrum = "Green"
    elif wavelength_nm < 590:
        spectrum = "Yellow"
    elif wavelength_nm < 620:
        spectrum = "Orange"
    elif wavelength_nm < 750:
        spectrum = "Red"
    elif wavelength_nm < 1e6:
        spectrum = "Infrared (IR)"
    else:
        spectrum = "Microwave/Radio"
    
    # Print results
    print(f"Mode n = {n} in cavity L = {L:.1f} nm:")
    print(f"  Wavelength: λ = {wavelength_nm:.1f} nm")
    print(f"  Energy: E = {energy_eV:.3f} eV")
    print(f"  Spectrum: {spectrum}")
    
    return {
        'wavelength_nm': wavelength_nm,
        'energy_eV': energy_eV,
        'spectrum': spectrum
    }


def minimize_rayleigh(L, N, phi0_fn, rq_fn, method="L-BFGS-B", maxiter=200000, ftol=1e-12, plot=True, penalty_weight=1e-2, jitter=1e-6):
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
        Function phi0_fn(x, L) -> array of shape (N,), initial guess satisfying φ(0)=φ(L)=0.
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
    phi_star : ndarray of shape (N,)
        Normalized converged mode profile.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Discretize domain
    x  = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    # Generate initial guess
    phi0 = np.asarray(phi0_fn(x, L), dtype=float)
    phi0[0]  = 0.0
    phi0[-1] = 0.0

    # Optimization variables (interior only)
    u0 = phi0[1:-1].copy()

    def objective(u_vec):
        phi = np.zeros_like(phi0)
        phi[1:-1] = u_vec
        
        # Rayleigh quotient from user (scale-invariant)
        R = rq_fn(phi, dx)
        # Soft constraint toward ||phi||_2^2 = 1 (under the grid measure)
        #den = float(np.sum(phi**2) * dx)
        return R #+ penalty_weight * (den - 1.0)**2

    # Run minimization
    res = minimize(
        objective,
        u0,
        method="L-BFGS-B",
        options={"maxiter": 200000, "maxfun": 200000, "ftol": 1e-12})
    
    # Rebuild and normalize converged wave
    phi_star = np.zeros_like(phi0)
    phi_star[1:-1] = res.x
    phi_star[0]  = 0.0
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

