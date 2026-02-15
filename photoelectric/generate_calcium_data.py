"""
Generate photoelectric I-V curves for calcium.
4 wavelengths × 4 intensities = 16 curves.

Students should find:
  - Stopping potential changes with wavelength/frequency
  - Stopping potential does NOT change with intensity
  - Saturation current scales with intensity
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Physical constants
h = 6.626e-34       # J·s
e = 1.602e-19       # C
c = 2.998e8          # m/s
h_eV = h / e         # eV·s

# Calcium
phi_ca = 2.87  # eV

# Wavelengths and intensities
wavelengths_nm = [200, 250, 300, 350]
intensity_labels = ["25%", "50%", "75%", "100%"]
saturation_currents_uA = [1.0, 2.0, 3.0, 4.0]


def photocurrent(v_retarding, v_stop, i_sat, sharpness=12.0):
    """Sigmoid I-V model with thermal broadening."""
    return i_sat / (1.0 + np.exp(sharpness * (v_retarding - v_stop)))


rows = []
for wl in wavelengths_nm:
    freq = c / (wl * 1e-9)
    v_stop = h_eV * freq - phi_ca

    # Voltage range: from -1 V (accelerating) to ~0.5 V past stopping potential
    v_min = -1.0
    v_max = v_stop + 0.5
    voltages = np.arange(v_min, v_max + 0.025, 0.05)

    for label, i_sat in zip(intensity_labels, saturation_currents_uA):
        currents = photocurrent(voltages, v_stop, i_sat)
        for v, i_ideal in zip(voltages, currents):
            noise = np.random.normal(0, 0.015 * i_sat + 0.003)
            i_measured = max(round(i_ideal + noise, 4), 0.0)
            rows.append({
                "wavelength_nm": wl,
                "frequency_e14_Hz": round(freq / 1e14, 4),
                "intensity": label,
                "retarding_voltage_V": round(v, 3),
                "photocurrent_uA": i_measured,
            })

df = pd.DataFrame(rows)
df.to_csv("/mnt/user-data/outputs/calcium_photoelectric_iv_curves.csv", index=False)

# Print summary
print("Calcium photoelectric I-V data")
print(f"Work function: {phi_ca} eV\n")
print(f"{'Wavelength (nm)':>16} {'Frequency (e14 Hz)':>20} {'V_stop (V)':>12}")
print("-" * 52)
for wl in wavelengths_nm:
    freq = c / (wl * 1e-9)
    vs = h_eV * freq - phi_ca
    print(f"{wl:>16} {freq/1e14:>20.4f} {vs:>12.3f}")

print(f"\nTotal rows: {len(df)}")
print(f"Curves: {len(wavelengths_nm)} wavelengths × {len(intensity_labels)} intensities = {len(wavelengths_nm)*len(intensity_labels)}")
print(f"\nIntensities: {intensity_labels}")
print(f"Saturation currents: {saturation_currents_uA} µA")
