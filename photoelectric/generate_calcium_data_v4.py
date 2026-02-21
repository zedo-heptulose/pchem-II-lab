"""
Generate photoelectric I-V curves for calcium.
4 wavelengths × 4 intensities = 16 curves.

Model: I = I_sat * (1 - exp(-alpha * max(0, V - V_stop))) * collection(V)

The exponential controls the sharp onset (and determines the extrapolated
stopping potential). The collection factor is a slowly-varying envelope
that increases from ~70% near V_stop to ~100% at positive voltages,
representing the gradual collection of electrons emitted at wide angles
or from deeper bands. Because collection(V) varies slowly near V_stop,
it doesn't affect the linear extrapolation — it just modifies the slope.

Sign convention: positive voltage = accelerating, negative = retarding.
"""

import numpy as np
import pandas as pd
import os
import glob

np.random.seed(42)

# Physical constants
h = 6.626e-34
e_charge = 1.602e-19
c = 2.998e8
h_eV = h / e_charge

# Calcium
phi_ca = 2.87  # eV

# Wavelengths and intensities
wavelengths_nm = [200, 250, 300, 350]
intensity_labels = ["25", "50", "75", "100"]
saturation_currents_uA = [1.0, 2.0, 3.0, 4.0]


def photocurrent(voltage, v_stop_neg, i_sat, alpha=1.5):
    """
    Exponential onset × slowly varying collection envelope.

    Near V_stop:
        I ≈ I_sat * collection(V_stop) * alpha * (V - V_stop)  [linear]
        Extrapolation to I=0 gives V = V_stop regardless of collection.

    Far from V_stop:
        I → I_sat * collection(V) → I_sat  [saturation at positive V]
    """
    x = np.maximum(voltage - v_stop_neg, 0.0)
    onset = 1.0 - np.exp(-alpha * x)

    # Collection efficiency: slowly ramps from ~0.7 to ~1.0
    # Centered at 0V, very gentle (sharpness=0.5)
    collection = 0.7 + 0.3 / (1.0 + np.exp(-0.5 * voltage))

    return i_sat * onset * collection


output_dir = "/mnt/user-data/outputs"

print("Calcium photoelectric I-V data")
print(f"Work function: {phi_ca} eV\n")

# Verify extrapolation
print(f"{'λ (nm)':>8} {'V_stop (V)':>12} {'Extrap (V)':>12} {'Error (V)':>12}")
print("-" * 48)

for wl in wavelengths_nm:
    freq = c / (wl * 1e-9)
    v_stop = h_eV * freq - phi_ca
    v_stop_neg = -v_stop
    voltages = np.arange(v_stop_neg - 1.0, 3.025, 0.05)

    # Test extrapolation with 100% intensity (noiseless)
    i_test = photocurrent(voltages, v_stop_neg, 4.0)
    # Fit line to rising region: 5-40% of max
    i_max = i_test.max()
    mask = (i_test > 0.05 * i_max) & (i_test < 0.40 * i_max)
    if mask.sum() >= 3:
        coeffs = np.polyfit(voltages[mask], i_test[mask], 1)
        v_extrap = -coeffs[1] / coeffs[0]
    else:
        v_extrap = float('nan')

    print(f"{wl:>8} {v_stop_neg:>12.3f} {v_extrap:>12.3f} {abs(v_extrap - v_stop_neg):>12.3f}")

    # Generate noisy data for all intensities
    for label, i_sat in zip(intensity_labels, saturation_currents_uA):
        currents = photocurrent(voltages, v_stop_neg, i_sat)
        rows = []
        for v, i_ideal in zip(voltages, currents):
            noise = np.random.normal(0, 0.015 * i_sat + 0.003)
            i_measured = max(round(i_ideal + noise, 4), 0.0)
            rows.append({
                "voltage_V": round(v, 3),
                "current_uA": i_measured,
            })
        df = pd.DataFrame(rows)
        fname = f"ca_{wl}nm_{label}intens.csv"
        df.to_csv(os.path.join(output_dir, fname), index=False)

print(f"\n16 files written to {output_dir}")
