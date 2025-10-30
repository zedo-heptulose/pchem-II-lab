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

def standing_animation(omega = 2, Nframes=240, interval_ms=40):
    """
    Generate and display an animation of a standing wave for a given frequency.

    Parameters
    ----------
    omega : float
        Angular frequency of the wave (rad/s). Determines how fast
        the standing wave oscillates in time. For this function,
        the wave speed c is assumed to be 1, so omega = k.

    Nframes : int, optional
        Number of frames in the animation. A higher number results
        in smoother motion but requires more computation. Default is 240.

    interval_ms : int, optional
        Time interval between animation frames in milliseconds.
        Controls the playback speed. Default is 40.
    """
    # -------- parameters you may adjust --------
    k = omega                            # spatial wavenumber
    x = np.linspace(-np.pi, np.pi, 700)  # centered spatial domain
    # ------------------------------------------

    
    # Traveling waves (sine => node at x=0 in the sum)
    E_r = lambda x,t: np.sin(k*x - omega*t)   # right-moving
    E_l = lambda x,t: np.sin(k*x + omega*t)   # left-moving
    E_sum = lambda x,t: E_r(x,t) + E_l(x,t)   # standing wave = 2 sin(kx) cos(ωt)
    
    # Build figure/axes
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
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
    phases = np.linspace(0.0, 2*np.pi, Nframes, endpoint=False)
    
    def init():
        # start with empty lines/points
        for ln in (line_r, line_l, line_sum):
            ln.set_data([], [])
        for pt in (point_r, point_l, point_sum):
            pt.set_data([], [])
        return line_r, line_l, line_sum, point_r, point_l, point_sum
    
    def animate(theta):
        # phase -> time
        t = theta / omega
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

def ring_wave_animation(R=1.0, a=0.18, m=4, c=1.0, Nframes=160, Nt=360):
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
        but increase rendering time and file size). Default is 160.

    Nt : int, optional
        Number of angular sampling points around the ring (higher values yield
        smoother curves but heavier computation). Default is 360.
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
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), subplot_kw={"aspect":"equal"})
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

