# Photoelectric Effect — Assignment Structure Summary

## Overview
Students analyze I–V curves from a photoelectric effect experiment on calcium to extract stopping potentials, determine Planck's constant and the work function via linear regression of V0 vs frequency.

**Total points: ~87** (coded tasks + written problems + final reflection)

---

## Part 1 — The Experiment and the Model (conceptual intro)

- **Problem 1** (10 pts) — 3 conceptual questions on stopping potential, applied voltage vs photon energy, and what equal stopping potentials imply.

- **Problem 2** (10 pts) — 3 questions on Python functions as physical models (measured→derived mapping, why functions beat manual work, sanity checks).

---

## Part 2 — Conversions as Functions

| Task | Points | What students do |
|------|--------|------------------|
| 2.1 | 1 pt | Use provided `wavelength_nm_to_thz()` function |
| 2.2 | 2 pts | **Write** `frequency_thz_to_energy_joule()` using E = hf |
| 2.3 | 2 pts | **Write** `make_light_properties_table()` returning a DataFrame with wavelength/frequency/energy columns |

- **Problem 3** (10 pts) — Predict which wavelengths eject electrons given Ca work function; explain why frequency (not wavelength) gives a linear model.

---

## Part 3 — I–V Curves for Calcium

Data files: `Ca_200nm`, `Ca_250nm`, `Ca_300nm`, `Ca_350nm`

| Task | Points | What students do |
|------|--------|------------------|
| 3A | 5 pts | Plot one I–V curve, identify where current→0 |
| 3.1 | (embedded) | **Write** `extract_stopping_potential(df, threshold)` — mask, polyfit, solve for x-intercept |
| 3.2 | 5 pts | Apply stopping potential function to all 4 Ca datasets, build results table |

- **Problem 4** (5 pts) — Interpret V0 trend with wavelength; check consistency with photoelectric equation; discuss missing zero-crossing.

---

## Part 4 — Calcium Results Table

| Task | Points | What students do |
|------|--------|------------------|
| 4.1 | — | Call `make_light_properties_table()` for Ca wavelengths |
| 4.2 | 5 pts | Compute V0 for each dataset using `extract_stopping_potential`, choose/justify threshold |
| 4.3 | 5 pts | Merge light properties + V0 into one DataFrame |

- **Problem 5** (5 pts) — 4 questions: energy/V0 trends, linearity check, cutoff wavelength, error sources.

---

## Part 5 — Photoelectric Equation and Quantization

Substantial theory section (5.1–5.3) covering work function, cutoff frequency, and linear model interpretation.

- **Problem 6** (5 pts) — Pre-fitting conceptual questions: universal vs material constants, intensity effects, V0=0 meaning.

### Coding activities:

| Task | Points | What students do |
|------|--------|------------------|
| 5.4a | 5 pts | Plot V0 vs frequency |
| 5.4b | 5 pts | Fit line (V0 = mf + b), extract h and phi from slope/intercept |
| 5.4c | 5 pts | Compute cutoff frequency f_c and cutoff wavelength |

- **Problem 7** (10 pts) — 5 questions: slope→h relationship and units, negative intercept interpretation, intensity below cutoff, negative V0 artifacts, comparison to accepted h.

---

## Final Reflection (ungraded/graded?)
5 essay-style questions covering: linearity vs quantization evidence, intensity independence, slope/intercept physical meaning, cutoff frequency, and metal-dependence of the graph.

---

## Key Functions Students Write
1. `frequency_thz_to_energy_joule(frequency_thz)` → energy in J
2. `make_light_properties_table(wavelengths_nm)` → DataFrame
3. `extract_stopping_potential(df, threshold)` → V0

## Data
- 4 CSV files: Ca at 200, 250, 300, 350 nm (voltage vs current)
- Filter wavelengths also referenced: 365, 405, 436, 546, 577 nm (for Part 2 examples)
