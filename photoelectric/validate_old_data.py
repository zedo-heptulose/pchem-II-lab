"""
Plot old_data/ Ca I-V curves (4 files) alongside the new data/ 100% intensity
curves for comparison. Also prints extracted V0 for both.
Run from the photoelectric/ directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

old_dir = os.path.join(os.path.dirname(__file__), 'old_data')
new_dir = os.path.join(os.path.dirname(__file__), 'data')

wavelengths = [200, 250, 300, 350]
threshold_old = 0.1   # pA
threshold_new = 0.05  # uA

fig, axes = plt.subplots(len(wavelengths), 2, figsize=(14, 12),
                         gridspec_kw={'wspace': 0.3, 'hspace': 0.35})

summary = []

for row, wl in enumerate(wavelengths):
    # --- OLD DATA ---
    ax_old = axes[row, 0]
    old_path = os.path.join(old_dir, f'Ca_{wl}nm.csv')
    df_old = pd.read_csv(old_path)
    V_old = df_old['Voltage (V)'].to_numpy()
    I_old = df_old['Current (pA)'].to_numpy()

    mask_old = I_old > threshold_old
    if mask_old.sum() >= 2:
        m_old, b_old = np.polyfit(V_old[mask_old], I_old[mask_old], 1)
        V0_old = -b_old / m_old
    else:
        m_old, b_old, V0_old = np.nan, np.nan, np.nan

    ax_old.scatter(V_old, I_old, s=20, zorder=3, alpha=0.7)
    V_line = np.linspace(V_old.min(), max(V_old.max(), V0_old + 0.5), 100)
    ax_old.plot(V_line, m_old * V_line + b_old, '-', color='red', lw=1.5)
    ax_old.axvline(V0_old, color='red', ls=':', alpha=0.5)
    ax_old.axhline(0, color='gray', lw=0.5)
    ax_old.set_xlabel('Voltage (V)')
    ax_old.set_ylabel('Current (pA)')
    ax_old.set_title(f'OLD  {wl} nm  (V0={V0_old:.2f} V)', fontsize=10)

    # --- NEW DATA (100% intensity) ---
    ax_new = axes[row, 1]
    new_path = os.path.join(new_dir, f'ca_{wl}nm_100intens.csv')
    df_new = pd.read_csv(new_path)
    V_new = df_new['voltage_V'].to_numpy()
    I_new = df_new['current_uA'].to_numpy()

    mask_new = I_new > threshold_new
    if mask_new.sum() >= 2:
        m_new, b_new = np.polyfit(V_new[mask_new], I_new[mask_new], 1)
        V0_new = -b_new / m_new
    else:
        m_new, b_new, V0_new = np.nan, np.nan, np.nan

    ax_new.scatter(V_new, I_new, s=10, zorder=3, alpha=0.7)
    V_line = np.linspace(V_new.min(), max(V_new.max(), V0_new + 0.5), 100)
    ax_new.plot(V_line, m_new * V_line + b_new, '-', color='red', lw=1.5)
    ax_new.axvline(V0_new, color='red', ls=':', alpha=0.5)
    ax_new.axhline(0, color='gray', lw=0.5)
    ax_new.set_xlabel('Voltage (V)')
    ax_new.set_ylabel('Current (µA)')
    ax_new.set_title(f'NEW  {wl} nm, 100%  (V0={V0_new:.2f} V)', fontsize=10)

    summary.append({
        'wavelength_nm': wl,
        'V0_old (V)': round(V0_old, 3),
        'V0_new_100pct (V)': round(V0_new, 3),
        'old_npts': len(V_old),
        'new_npts': len(V_new),
        'old_unit': 'pA',
        'new_unit': 'uA',
        'old_V_range': f'{V_old.min():.2f} - {V_old.max():.2f}',
        'new_V_range': f'{V_new.min():.2f} - {V_new.max():.2f}',
    })

fig.suptitle('Old Data vs New Data (100% intensity) — Side by Side', fontsize=14)
fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'validation_old_vs_new.png'), dpi=150)
plt.show()

print("\nComparison summary:")
df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

print("\nKey differences:")
print("  - Old data: columns 'Voltage (V)', 'Current (pA)'; includes Trial, Metal, Frequency, Wavelength cols")
print("  - New data: columns 'voltage_V', 'current_uA'; no metadata columns")
print("  - Old data: voltage starts near 0 V (positive only)")
print("  - New data: voltage starts at -1.0 V (includes negative/forward bias region)")
print("  - Old data: ~20 points per file")
print("  - New data: ~100 points per file")
print("  - Units: old=picoamps, new=microamps")
print("\nSaved validation_old_vs_new.png")
