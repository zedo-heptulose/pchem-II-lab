import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt



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
    
    
def interactive_superposition_plot():
    widgets.interact(superposition_plot, f1=(0,5,0.1), p1=(0,20,0.1),f2=(0,5,0.1),p2=(0,20,0.1))


def waves_and_superposition(f1,p1,f2,p2):
    #creating list of x values for use with our functions
    x = np.linspace(0, 10, 640) 
    #transforming to angular frequencies for use in sine
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    #write what comes after!
    y_1 = np.sin(w1 * x + p1)
    y_2 = np.sin(w2 * x + p2)
    y_superposition = y_1 + y_2
        
    return x, y_1, y_2, y_superposition

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


from matplotlib.gridspec import GridSpec
def interference_plot_template(theta=0,d=1000,wavelength = 400,iterations=2,crystal=False,extra_vars=[]):
    fig = plt.figure(figsize=(6,6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    ax2 = fig.add_subplot(gs[1, :])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax_text = fig.add_subplot(gs[0, 0])

    if d != 0:
        ax1.set_xlim([-d * 2, d* 2])
        ax1.set_ylim([0, d * 3]) 
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f'Path of parallel rays, $\\theta={np.degrees(theta):.1f}\\degree$')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')

    if crystal:
        L = 2 * d * np.sin(theta)
    else:
        L = d * np.sin(theta)
    phase = L / wavelength
    if iterations > 2:
        orders = np.array([i for i in range(1,iterations)])
        phases = orders * phase
        ax2 = plot_wave_interference(phases,wavelength,ax2)
    else:
        ax2 = plot_wave_interference([phase],wavelength,ax2)
        
    ax2.set_xlim(-wavelength * 2.5,wavelength * 2.5)
    ax2.set_ylim(-1.05,1.05)
    ax2.set_title(f'Interference of parallel rays')
    ax2.set_xlabel('Position (nm)')
    ax2.set_ylabel('Amplitude (not to scale)')

    ax_text.axis("off") 
    variable_text = [
        f'$\\lambda={wavelength}nm$',
        f'$\\theta={np.degrees(theta):.1f}\\degree$',
    ]
    if crystal:
        variable_text.append(f'$d={d:.3f}nm$')
    else:
        variable_text.append(f'$d={d:.0f}nm$')
    if extra_vars:
        variable_text.extend(extra_vars)
    if crystal:
        variable_text.append(f'$L=2d\\cdot{{}}sin(\\theta) = {L:.3f}nm$')
        variable_text.append(f'$2\\theta={2*np.degrees(theta):.1f}\\degree$')
    else:
        variable_text.append(f'$L=d\\cdot{{}}sin(\\theta) = {L:.0f}nm$')
    ax_text.text(0.25, 0.5, "\n\n".join(variable_text), fontsize=10, ha="center", va="center")
    plt.subplots_adjust(hspace=0.4)

    return fig, (ax1,ax2,ax_text)



def interference_subplot(ax1,theta,d,iterations):
    signed_angle = theta
    theta = np.abs(theta)
    height = d * 3
    translations = range(-iterations//2,iterations//2)
    phase_lengths = []
    for i, translation in enumerate(translations):
        origin = np.array([float(translation) * d,0.00])
        p1 = origin + np.array([-d/2 , 0.0])
        p2 = origin + np.array([d/2, 0.0])
        p3_x = d * np.cos(theta) * np.cos(theta)
        p3_y = d * np.cos(theta) * np.sin(theta)
        p3 = p1 + np.array([p3_x,p3_y])
        p45_trans = np.array([-np.tan(theta) * height,height])
        p4 = p2 + p45_trans
        points = [p1,p2,p3,p5]
        x = np.array([point[0] for point in points])
        y = np.array([point[1] for point in points])
        phase_length = np.linalg.norm(p2-p3)
        phase_lengths.append(phase_length)

        if i == 0:
            pairs = [(p2,p4)]
        else:
            pairs = [(p1,p3),(p2,p3),(p3,p4)]
        for pair in pairs:
            x = np.array([point[0] for point in pair if type(point) is np.ndarray])
            y = np.array([point[1] for point in pair if type(point) is np.ndarray])

            if signed_angle > 0:
                ax1.plot(-x,y)
            else:
                ax1.plot(x,y)
                

