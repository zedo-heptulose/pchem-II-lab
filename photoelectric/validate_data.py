"""
Validate photoelectric data by running the full notebook analysis pipeline.
Produces three figures:
  1. All 16 I-V curves (4x4 grid)
  2. V0 vs frequency and V0 vs intensity
  3. Final linear fit: electron KE vs frequency (extracts slope ≈ h)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light, Planck, e as elementary_charge
from scipy.stats import linregress

# --- Functions (same as notebook) ---

def wavelength_nm_to_thz(wavelength):
    frequency_hz = speed_of_light / (wavelength * 1e-9)
    frequency_thz = frequency_hz * 1e-12
    return frequency_thz

def linear_fit(x, y):
    result = linregress(x, y)
    return result.slope, result.intercept

def get_V0(ca_data):
    V = ca_data['voltage_V']
    I = ca_data['current_uA']
    sat_current = max(I)
    sat_filter = I < (sat_current * 0.5)
    zero_filter = I > (sat_current * 0.2)
    total_filter = sat_filter & zero_filter
    V_filter = V[total_filter]
    I_filter = I[total_filter]
    slope, intercept = linear_fit(V_filter, I_filter)
    return -intercept / slope

def get_filename(wavelength, intensity):
    return f'data/ca_{wavelength}nm_{intensity}intens.csv'

# --- Config ---
wavelengths = [200, 250, 300, 350]
intensities = [25, 50, 75, 100]

# ============================================================
# Figure 1: All 16 I-V curves with filtered region + linear fit
# ============================================================
fig1, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=False, sharey=False)
fig1.suptitle('All 16 I-V Curves with Filtered Linear Fits', fontsize=14)

for i, wave in enumerate(wavelengths):
    for j, intens in enumerate(intensities):
        ax = axes[i][j]
        fname = get_filename(wave, intens)
        ca_data = pd.read_csv(fname)
        V = ca_data['voltage_V']
        I = ca_data['current_uA']

        # Filter
        sat_current = max(I)
        sat_filter = I < (sat_current * 0.5)
        zero_filter = I > (sat_current * 0.2)
        total_filter = sat_filter & zero_filter
        V_f = V[total_filter]
        I_f = I[total_filter]

        # Fit
        slope, intercept = linear_fit(V_f, I_f)
        V0 = -intercept / slope
        I_fit = slope * V_f + intercept

        ax.scatter(V, I, s=4, alpha=0.4, label='all data')
        ax.scatter(V_f, I_f, s=6, color='orange', label='filtered')
        ax.plot(V_f, I_fit, color='red', linewidth=1.5)
        ax.axvline(V0, color='green', linestyle='--', linewidth=1, label=f'V0={V0:.2f}')
        ax.set_title(f'{wave} nm, {intens}%', fontsize=9)
        if i == 3:
            ax.set_xlabel('V')
        if j == 0:
            ax.set_ylabel('I (µA)')

fig1.tight_layout()
fig1.savefig('validation_iv_curves.png', dpi=150)
print('Saved: validation_iv_curves.png')

# ============================================================
# Figure 2: V0 vs frequency and V0 vs intensity
# ============================================================
# Collect all V0 values
rows = []
for wave in wavelengths:
    for intens in intensities:
        ca_data = pd.read_csv(get_filename(wave, intens))
        V0 = get_V0(ca_data)
        rows.append({'wavelength_nm': wave, 'intensity': intens,
                      'frequency_thz': wavelength_nm_to_thz(wave), 'V0': V0})

data = pd.DataFrame(rows)
print('\nResults table:')
print(data.to_string(index=False))

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for intens in intensities:
    sub = data[data['intensity'] == intens]
    ax1.plot(sub['frequency_thz'], sub['V0'], 'o-', label=f'{intens}%')
ax1.set_xlabel('Frequency (THz)')
ax1.set_ylabel('Stopping Potential V0 (V)')
ax1.set_title('V0 vs Frequency (by intensity)')
ax1.legend()

for wave in wavelengths:
    sub = data[data['wavelength_nm'] == wave]
    ax2.plot(sub['intensity'], sub['V0'], 's-', label=f'{wave} nm')
ax2.set_xlabel('Intensity (%)')
ax2.set_ylabel('Stopping Potential V0 (V)')
ax2.set_title('V0 vs Intensity (by wavelength)')
ax2.legend()

fig2.tight_layout()
fig2.savefig('validation_v0_trends.png', dpi=150)
print('Saved: validation_v0_trends.png')

# ============================================================
# Figure 3: Electron KE vs frequency — extract Planck's constant
# ============================================================
data_100 = data[data['intensity'] == 100].copy()
data_100['f_hz'] = data_100['frequency_thz'] * 1e12
data_100['e_KE'] = -elementary_charge * data_100['V0']

slope, intercept = linear_fit(data_100['f_hz'], data_100['e_KE'])
f_fit = np.linspace(data_100['f_hz'].min() * 0.95, data_100['f_hz'].max() * 1.05, 100)
KE_fit = slope * f_fit + intercept

fig3, ax = plt.subplots(figsize=(8, 5))
ax.scatter(data_100['f_hz'], data_100['e_KE'], s=60, zorder=5, label='data (100%)')
ax.plot(f_fit, KE_fit, 'r-', label='linear fit')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Electron Kinetic Energy (J)')
ax.set_title('Electron KE vs Frequency — Extracting α')
ax.legend()

# Annotate
work_fn_eV = -intercept / elementary_charge
ax.text(0.05, 0.95,
        f'slope (α) = {slope:.3e} J·s\n'
        f'Planck h   = {Planck:.3e} J·s\n'
        f'error      = {abs(slope - Planck)/Planck*100:.2f}%\n'
        f'W_ex       = {work_fn_eV:.3f} eV',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig3.tight_layout()
fig3.savefig('validation_planck.png', dpi=150)
print('Saved: validation_planck.png')

print('\n--- Summary ---')
print(f'Extracted slope (α): {slope:.4e} J·s')
print(f'Known Planck h:      {Planck:.4e} J·s')
print(f'Relative error:      {abs(slope - Planck)/Planck*100:.2f}%')
print(f'Work function:       {work_fn_eV:.3f} eV')
