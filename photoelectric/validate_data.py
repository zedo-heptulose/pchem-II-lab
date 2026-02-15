"""
Quick validation: plot all 16 Ca I-V curves with filtered linear fits and extracted V0.
Layout: 4 rows (wavelengths) x 4 cols (intensities).
Run from the photoelectric/ directory.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(__file__), 'data')
files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))

# parse wavelength and intensity from filenames like ca_200nm_25intens.csv
def parse_fname(fname):
    m = re.match(r'ca_(\d+)nm_(\d+)intens\.csv', fname)
    return int(m.group(1)), int(m.group(2))

wavelengths = sorted(set(parse_fname(f)[0] for f in files))
intensities = sorted(set(parse_fname(f)[1] for f in files))

# build lookup: (wavelength, intensity) -> filename
lookup = {parse_fname(f): f for f in files}

threshold = 0.05  # uA

fig, axes = plt.subplots(len(wavelengths), len(intensities),
                         figsize=(16, 12), sharex=True, sharey='row')

v0_table = []

for row, wl in enumerate(wavelengths):
    for col, intens in enumerate(intensities):
        ax = axes[row, col]
        fname = lookup.get((wl, intens))
        if fname is None:
            ax.set_visible(False)
            continue

        df = pd.read_csv(os.path.join(data_dir, fname))
        V = df['voltage_V'].to_numpy()
        I = df['current_uA'].to_numpy()

        # filtered fit
        mask = I > threshold
        if mask.sum() >= 2:
            m, b = np.polyfit(V[mask], I[mask], 1)
            V0 = -b / m
        else:
            m, b, V0 = np.nan, np.nan, np.nan

        v0_table.append({'wavelength_nm': wl, 'intensity_pct': intens, 'V0': V0})

        # plot
        ax.scatter(V, I, s=10, zorder=3, alpha=0.7)
        V_line = np.linspace(V.min(), V.max(), 100)
        ax.plot(V_line, m * V_line + b, '-', color='red', lw=1.5)
        ax.axvline(V0, color='red', ls=':', alpha=0.5)
        ax.axhline(0, color='gray', lw=0.5)

        ax.set_title(f'{wl} nm, {intens}%  (V0={V0:.2f} V)', fontsize=9)
        if row == len(wavelengths) - 1:
            ax.set_xlabel('Voltage (V)')
        if col == 0:
            ax.set_ylabel('Current (uA)')

fig.suptitle('Photoelectric Effect — All Ca I-V Curves (filtered fit)', fontsize=14)
fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'validation_plots.png'), dpi=150)
plt.show()

# print V0 summary table
v0_df = pd.DataFrame(v0_table)
v0_pivot = v0_df.pivot(index='wavelength_nm', columns='intensity_pct', values='V0')
print("\nExtracted V0 (volts) — rows=wavelength, cols=intensity%:")
print(v0_pivot.round(3).to_string())
print("\nV0 std dev across intensities (should be small if intensity doesn't matter):")
print(v0_pivot.std(axis=1).round(4).to_string())
print("\nSaved validation_plots.png")
