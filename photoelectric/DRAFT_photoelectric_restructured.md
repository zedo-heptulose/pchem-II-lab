# The Photoelectric Effect

In this experiment, you will investigate the photoelectric effect as a proof of the particle nature of light.

By measuring the stopping potential for photoelectrons emitted under illumination by light of different wavelengths, you will connect experimentally measured voltages to fundamental physical constants.

As in the previous X-ray diffraction lab, the emphasis of this assignment is not only on generating plots, but on understanding how mathematical and computational models represent physical relationships.

> SOURCE: worked version title cell, trimmed

---

**Chemistry Learning Objectives:**
 - C3.1: Apply the photoelectric effect to extract physical constants.
 - C3.2: Explain how the photoelectric effect suggests the existence of photons.
 - C3.3: Interpret experimental trends using physical reasoning.

**Programming Learning Objectives:**
- P3.1: Interpret Python functions as physical models.
- P3.2: Use functions to transform experimental data.
- P3.3: Analyze and visualize data using modular code.
- P3.4: Use `pandas` to manipulate and filter tabular (spreadsheet) data.
- P3.5: Perform linear regression using `NumPy`.

> SOURCE: worked version. These already incorporate TODO changes (added P3.4, P3.5, reworded C3.2).

---
---

# Warmup — Functions and Abstraction

> SOURCE: worked version "Warmup" cell, verbatim.

In this notebook, we will use **Python functions** to represent relationships between physical quantities. A Python function is a rule that takes one or more **inputs** and returns an **output**. In the context of scientific computing, the most important point is that a function is not "just code": it is a *model* for a relationship we believe is well-defined.

A function has three parts:

1. **Inputs (arguments):** the quantities you provide to the function.
2. **The rule:** the operations that transform inputs into an output.
3. **Output (return value):** the quantity the function produces.

For example, if `lam_nm` is a wavelength in nanometers, a conversion function might map

$$
\lambda \ \mapsto\ f
$$

meaning: "given a wavelength, return the corresponding frequency." In Python, this looks like a function that takes a number (or an array of numbers) and returns a number (or array).

In this lab, we will repeatedly use functions to carry out transformations that mirror the analysis steps you perform conceptually:

1. **Convert light properties**
   $$
   \lambda \ \mapsto\ f, \qquad f \ \mapsto\ E_{\text{photon}}
   $$
2. **Extract a stopping potential from an I–V dataset**
   $$
   \{(V, I)\} \ \mapsto\ V_0
   $$
3. **Use a linear model to extract constants**
   $$
   f \ \mapsto\ V_0 \quad \Rightarrow \quad \text{slope and intercept}
   $$

A major advantage of functions is that they allow you to apply the *same* rule consistently across many datasets. For example, once you have a reliable function for extracting $V_0$ from a current–voltage curve, you can apply it to every wavelength without repeating manual steps. This reduces errors and makes your analysis reproducible.

Finally, throughout this notebook you should be able to answer the following about any function you use:

1. What are the **inputs**, including their **units**?
2. What is the **output**, including its **units**?
3. What physical relationship or modeling assumption does the function encode?
4. How could you check that the output is reasonable (order of magnitude, limiting behavior, trends)?

---

## Problem 1
`10 points`
- P3.1: Interpret Python functions as physical models.
- P3.2: Use functions to transform experimental data.

a) In this experiment, give one example of a function that maps a *measured* quantity to a *derived* quantity.

b) Why is it useful to package a repeated analysis step (such as determining $V_0$) into a function instead of doing it manually for each wavelength?

c) Suppose a function returns a stopping potential with the wrong sign. What is one physical check you could use to notice that something is inconsistent?

> SOURCE: was Problem 2 in both new and worked versions. Renumbered.

---

## Warmup Coding Activity — Conversions as Functions

> SOURCE: Section 2 from new/scratch versions, lightly rearranged.

In the photoelectric effect, the properties of light are most naturally described in terms of **wavelength**, **frequency**, and **energy**. While wavelength is the quantity selected experimentally using optical filters, the photoelectric equation is written in terms of frequency. As a result, the first step in the analysis is to convert between these representations in a consistent and transparent way.

In this part of the notebook, you will use Python functions to carry out these conversions. The goal is not simply to compute numerical values, but to understand how each function encodes a physical relationship and how the output of one function becomes the input to the next.

---

### W.1 — Converting Wavelength to Frequency

`1 point`

The frequency $f$ of light is related to its wavelength $\lambda$ by the speed of light $c$:

$$
f = \frac{c}{\lambda}
$$

In the laboratory, wavelengths are specified in **nanometers**, while frequency is typically expressed in **hertz (s$^{-1}$)**. A conversion function allows us to apply this relationship consistently for every wavelength used in the experiment.

In the code cell below, you will use a provided Python function that converts a wavelength in nanometers to a frequency in terahertz.

```python
from scipy.constants import speed_of_light # in m/s

def wavelength_nm_to_thz(wavelength):
    """
    Convert a wavelength in nanometers to frequency in terahertz.

    Parameters
    ----------
    wavelength : float or array-like
        Wavelength of light in nanometers (nm).

    Returns
    -------
    frequency : float or array-like
        Frequency of light in terahertz (THz).
    """
    # speed_of_light is in m/s.
    # wavelength is in nm = 1e-9 m.
    # dividing gives Hz; multiplying by 1e-12 gives THz.
    # net factor: 1e-9 * 1e-12 = 1e-3 on c, divided by wavelength.
    frequency = speed_of_light * 1e-3 / wavelength
    return frequency
```

> SOURCE: Section 2.1 from new/scratch. Added comment explaining the 1e-3 factor per TODO item "mysterious 10**-3 in frequency function."
> CLAUDE ADDITION: the inline comments explaining the unit conversion.

---

### W.2 — Photon Energy as a Function of Frequency

`2 points`

Once the frequency of the incident light is known, the energy of an individual photon is given by

$$
E = h f
$$

where $h$ is Planck's constant. Although Planck's constant will later be *determined experimentally* in this lab, it is still useful at this stage to compute photon energies in order to compare different wavelengths qualitatively.

Write a function that takes frequency in terahertz (THz) and returns photon energy in joules (J).

```python
import numpy as np
from scipy.constants import Planck # h in J/Hz

def frequency_thz_to_energy_joule(frequency_thz):
    """
    Convert frequency (THz) to photon energy (J).

    Parameters
    ----------
    frequency_thz : float or array-like
        Frequency in terahertz (THz).

    Returns
    -------
    energy_joule : float or array-like
        Photon energy in joules (J).
    """
    # YOUR CODE HERE
    pass

# Quick checks
print(frequency_thz_to_energy_joule(822))  # order of magnitude ~ 1e-19 J
print(frequency_thz_to_energy_joule(np.array([822, 740])))
```

> SOURCE: Section 2.2 from new/scratch.

---

### W.3 — Light Properties Table

`2 points`

In the rest of the notebook you will repeatedly need wavelength, frequency, and photon energy for the same set of filters. Rather than recomputing these values manually each time, write a function that takes a list/array of wavelengths (nm) and returns a table containing:

- wavelength (nm)
- frequency (THz)
- photon energy (J)

Your function should call your functions from Parts W.1 and W.2.

```python
import pandas as pd

def make_light_properties_table(wavelengths_nm):
    """
    Build a table of wavelength (nm), frequency (THz), and photon energy (J).

    Parameters
    ----------
    wavelengths_nm : array-like
        Wavelengths in nanometers (nm).

    Returns
    -------
    table : pandas.DataFrame
        Columns: wavelength_nm, frequency_thz, energy_joule
    """
    wavelengths_nm = np.array(wavelengths_nm)

    # YOUR CODE HERE
    pass

filters_nm = [200, 250, 300, 350]
make_light_properties_table(filters_nm)
```

> SOURCE: Section 2.3 from new/scratch. Changed filter wavelengths to match Ca datasets (was [365,405,436,546,577]).

---
---

# Part 1 — The Experiment and Processing One Spectrum

> SOURCE: worked version Part 1 intro + Part 3A, rearranged.

## 1.1 — Experimental Measurements

The photoelectric effect was one of the pivotal proofs of quantum mechanics and led to Albert Einstein's discovery of the photon. How can such a simple experiment give us that much information?

In the photoelectric effect experiment, we shine monochromatic light on a metal cathode inside a vacuum photocell. The light can excite electrons in the metal and cause them to be emitted from the surface. The emitted electrons are collected at the anode, resulting in a measurable current.

<figure style="text-align: center;">
  <img src="https://cdn1.byjus.com/wp-content/uploads/2019/09/Experimental-study-of-photoelectric-effect.png"
       width="600">
  <figcaption style="font-size: 0.9em; margin-top: 6px;">
    Image credit: <a href="https://byjus.com/jee/photoelectric-effect/" target="_blank">
    https://byjus.com/jee/photoelectric-effect/
    </a>
  </figcaption>
</figure>

For a fixed wavelength, the applied voltage across the photocell is varied while the resulting current is recorded, producing a current–voltage (I–V) curve.

<figure style="text-align: center;">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRL1TvWZUgrBtaUMI7v-4Lj-T_jkILHP9yJHA&s"
       width="600">
  <figcaption style="font-size: 0.9em; margin-top: 6px;">
    Image credit: <a href="https://www.thestudentroom.co.uk/showthread.php?t=3318259" target="_blank">
    https://www.thestudentroom.co.uk/showthread.php?t=3318259
    </a>
  </figcaption>
</figure>

The applied voltage *resists* the travel of the electrons, so for an electron to complete the circuit, it must have enough kinetic energy to overcome the potential between the anode and cathode. This results in less and less current as we increase the voltage (why?).

So, this experiment is able to give us a measure of how much energy the light is giving to the electrons. But how does this suggest the existence of quantum mechanics? Keep going and find out!

> SOURCE: worked version Part 1 intro cell, verbatim.
> NOTE per TODO: figures need replacing (not Byjus). Placeholder for now.

---

## 1.2 — Visualizing One I–V Dataset

`10 points`

Select one dataset (for example, `Ca_200nm`) and generate a plot of:

- Current (pA) on the y-axis
- Voltage (V) on the x-axis

Your plot should include:
- Clearly labeled axes with units,
- A descriptive title including the wavelength,
- A linear regression of your data set

```python
import matplotlib.pyplot as plt
import pandas as pd

# some helpful functions:
# plt.scatter(x,y), plt.plot(x,y), plt.title(title), plt.xlabel(text), plt.ylabel(text)
# in general, use plt.scatter() for experimental data and plt.plot() for regressions

Ca_200nm = pd.read_csv('data/Ca_200nm.csv')

# YOUR CODE HERE
```

> SOURCE: worked version Part 3A, adjusted. Matplotlib cheatsheet added per TODO.

---

<!-- CLAUDE ADDITION: This sub-section is new. The TODO says: "let the students make the mistake of
taking the linear regression without filtering first and ask what goes wrong. Ask them to plot their
linear regression before and after filtering." and "separate filtering data and doing the linear
regression into multiple steps." -->

## 1.3 — Fitting the Data: Why Filtering Matters

`5 points`

You just fit a line to the *entire* I–V dataset. Look at your plot.

a) Does your regression line accurately represent the region where the current transitions from positive to zero? Why or why not?

b) The flat region at high voltage (where current ≈ 0) contains no useful information about the stopping potential, but it *does* affect the regression. Re-do your fit using only the data points where the current is above some small threshold (e.g., `I > 0.1` pA). Plot both the old and new regression lines on the same figure.

```python
# Step 1: Extract voltage and current arrays
V = Ca_200nm['Voltage (V)'].to_numpy()
I = Ca_200nm['Current (pA)'].to_numpy()

# Step 2: Create a boolean mask to select the fitting region
# YOUR CODE HERE

# Step 3: Apply the mask
# YOUR CODE HERE

# Step 4: Fit a line using np.polyfit(V_fit, I_fit, 1)
# YOUR CODE HERE

# Step 5: Plot original data, unfiltered fit, and filtered fit
# YOUR CODE HERE
```

c) The stopping potential $V_0$ is the voltage where the current reaches zero. Using your filtered fit $I = mV + b$, solve for the voltage where $I = 0$:

$$V_0 = -\frac{b}{m}$$

```python
# YOUR CODE HERE
```

**Sanity check:** Does the trend in the data match the expected physical behavior? Does your $V_0$ land near where the current visually crosses zero on your plot?

> CLAUDE ADDITION: entire section 1.3. Implements TODO items about separating filtering and regression,
> letting students see the mistake, and framing the answer as a "useful sanity check."

---

## Problem 2

`5 points`

> SOURCE: adapted from Problem 1 in new version + TODO additions.

a) Why is the stopping potential associated with the maximum kinetic energy of the emitted electrons rather than the average kinetic energy?

b) For a fixed wavelength of light, why does changing the applied voltage affect the measured current but not the energy of individual photons?

> NOTE: Problem 1c from the original ("if two wavelengths produced the same stopping potential...")
> removed to stay at max 3 SRQs. Could be re-added if desired.

<!-- CLAUDE ADDITION: reduced from 3 to 2 questions per TODO "max 3 SRQs per part" and moved to
follow the coding activity per TODO "short response questions should follow a programming activity." -->

---
---

# Part 2 — Abstracting to a Function and Processing All Spectra

> SOURCE: Parts 3B and 3.2/4.2 from worked/scratch versions, rearranged.
> This implements the TODO: "make students do all the steps separately and then abstract the whole
> sequence as a function" and "explicitly talk about abstraction here."

## 2.1 — From Steps to Function

In Part 1, you extracted the stopping potential from a single dataset by performing four steps manually:

1. Extract voltage and current arrays
2. Mask to a fitting region
3. Fit a line with `np.polyfit`
4. Solve for the x-intercept

You will now need to repeat this exact procedure for every dataset. Rather than copy-pasting the same code four times, you can **package these steps into a function** — a reusable block of code that performs the same operation on any input.

This is an example of **abstraction**: taking a concrete sequence of steps and wrapping it into a single named operation. The details are hidden inside the function; the user only needs to know *what goes in* and *what comes out*.

$$
\{(V, I)\}_{\lambda} \mapsto V_0(\lambda)
$$

<!-- CLAUDE ADDITION: the "abstraction" framing paragraph above. The function mapping notation
is from the scratch/new version 3B. -->

Write a function called `extract_stopping_potential` that takes a DataFrame and a threshold value, and returns $V_0$.

```python
def extract_stopping_potential(df, threshold):
    """
    Estimate the stopping potential V0 from a current–voltage dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing 'Voltage (V)' and 'Current (pA)' columns.
    threshold : float
        Current cutoff (in pA) used to define the fitting region.

    Returns
    -------
    V0 : float
        Estimated stopping potential in volts.
    """
    # YOUR CODE HERE
    # (Hint: same steps you did in Part 1.3, but now inside a function)
    pass

# Test it on your Ca_200nm data — should match what you got in Part 1
V0_200 = extract_stopping_potential(Ca_200nm, threshold=0.1)
print(V0_200)
```

> SOURCE: function skeleton from new/worked/scratch versions (all identical). Simplified
> by removing the step-by-step skeleton per TODO "want to make students write their own
> functions from scratch without giving them skeletons."

---

## 2.2 — Apply to All Calcium Datasets

`5 points`

Now apply the same stopping potential function to all four datasets:

- `Ca_200nm`
- `Ca_250nm`
- `Ca_300nm`
- `Ca_350nm`

> SOURCE: Section 3.2 from worked version.

```python
threshold = None  # TODO: choose a threshold value appropriate for your current units

import os
files = sorted(os.listdir('./data/'))

V0_values = []
for file in files:
    path = './data/' + file
    # YOUR CODE HERE: read csv, extract V0, append to V0_values

V0_values
```

> SOURCE: Section 4.2 code cell from worked/scratch versions.

---

## 2.3 — Build the Results Table

`5 points`

Combine your light properties table from the Warmup with your stopping potentials into a single results table.

Your final table must include the columns:
- `wavelength_nm`
- `frequency_thz`
- `energy_joule`
- `V0_volt`

<!-- CLAUDE ADDITION: short pandas primer below, per TODO "need a primer on adding columns." -->

**Quick `pandas` reference — adding a column:**
```python
# If you have a DataFrame called `df` and a list called `values`:
df["new_column_name"] = values
```

```python
wavelengths_nm = [200, 250, 300, 350]
light_table = make_light_properties_table(wavelengths_nm)

# YOUR CODE HERE: add V0_volt column
results = light_table.copy()
results["V0_volt"] = None  # replace None with your V0_values

results
```

> SOURCE: Sections 4.1 + 4.3 from worked/scratch versions, merged.

---

## Problem 3

`5 points`

> SOURCE: Problem 4 from new/worked versions, renumbered. Reduced from original Problem 5
> (4 questions) to 3 per TODO.

a) How does the stopping potential change as wavelength increases from 200 nm to 350 nm?

b) Does this trend agree with the expectation from the photoelectric equation? Explain.

c) If the 350 nm dataset showed no zero-crossing, what would that imply about the relationship between photon energy and the work function of calcium?

---
---

# Part 3 — What Matters: Wavelength vs. Intensity

<!-- CLAUDE ADDITION: This entire section is new. It implements the TODO item:
"If we want to present it as a proof of quantum mechanics we need to have them process
signals at different intensities and see that nothing changes."

NO INTENSITY DATA EXISTS YET in data/. This section is a placeholder that will need
data files (e.g., Ca_200nm_low.csv, Ca_200nm_high.csv) to be created or sourced.
-->

> **STATUS: PLACEHOLDER — needs intensity-varied data files.**

In Parts 1–2, you varied the *wavelength* of light and observed that the stopping potential changed. But what happens if you keep the wavelength fixed and change the *intensity* (brightness) of the light instead?

This question is at the heart of why the photoelectric effect was so revolutionary. Classical wave theory predicts that brighter light should deliver more energy to the electrons. Under that model, increasing intensity at any wavelength should eventually eject electrons and increase their kinetic energy. Quantum theory makes a very different prediction.

## 3.1 — Comparing I–V Curves at Different Intensities

`5 points`

Load the following datasets, which were collected at the same wavelength but different light intensities:

- `Ca_200nm_low.csv`   ← **FILE NEEDED**
- `Ca_200nm_high.csv`  ← **FILE NEEDED**

Plot both I–V curves on the same figure. Use the same axis labels and include a legend indicating which curve corresponds to which intensity.

```python
# YOUR CODE HERE
```

## 3.2 — Extracting and Comparing Stopping Potentials

`5 points`

Apply your `extract_stopping_potential` function to both datasets. Record $V_0$ for each.

```python
# YOUR CODE HERE
```

## Problem 4

`10 points`

a) When you increase the intensity at the same wavelength, does the stopping potential change? What about the maximum current?

b) Classical wave theory predicts that increasing intensity should increase the kinetic energy of emitted electrons. Does your data support this prediction? Explain.

c) Explain in your own words how this result — combined with the wavelength dependence from Part 2 — supports the idea that light energy comes in discrete packets (photons).

<!-- END CLAUDE ADDITION -->

---
---

# Part 4 — Discovering a Fundamental Constant

<!-- CLAUDE ADDITION: The framing of this section is new — per TODO "Would prefer students
'discover' Planck's constant for themselves here" and "don't tell them it's Planck's, let
them find out." The theory text below is pulled from worked version section 5.1–5.3 but
trimmed and reframed so h is not named up front. -->

You have now established two experimental facts:

1. The stopping potential depends on the *frequency* of the incident light.
2. The stopping potential does *not* depend on the *intensity* of the incident light.

These observations are inconsistent with classical wave theory, which predicts that more intense light should produce more energetic electrons at any frequency. Instead, the data suggest that light delivers energy in fixed amounts that depend only on frequency. In this section, you will determine *how* $V_0$ depends on frequency, and extract the proportionality constant from your data.

## 4.1 — The Linear Model

> SOURCE: Section 5.1–5.3 from worked version, condensed.

When light of frequency $f$ strikes a metal surface, electrons are emitted only if the photon energy exceeds a material-dependent threshold called the **work function** $\phi$. Conservation of energy gives:

$$
eV_0 = hf - \phi
$$

or equivalently:

$$
V_0 = \frac{h}{e} f - \frac{\phi}{e}
$$

This predicts a **linear** relationship between $V_0$ and $f$, where:
- The **slope** $m = h/e$ is related to a universal constant
- The **intercept** $b = -\phi/e$ is related to a property of the metal

> NOTE: The full theory discussion from sections 5.1 (Fermi energy, vacuum level, etc.)
> and 5.2 (cutoff frequency, classical vs quantum) is available in the worked version.
> It was trimmed here per TODO "theory stuff yaps a bit — make it more concise and listed."
> Decide how much to keep.

---

## 4.2 — Plot $V_0$ vs. Frequency

`5 points`

> SOURCE: Section 5.4 code cell from worked version.

```python
import matplotlib.pyplot as plt

f_thz = results["frequency_thz"].to_numpy()
V0 = results["V0_volt"].to_numpy()

# YOUR CODE HERE: scatter plot with axis labels and title
```

---

## 4.3 — Fit the Linear Model

`5 points`

> SOURCE: Section 5.4 second code cell from worked version.

```python
from scipy.constants import e  # elementary charge in coulombs

# Convert frequency from THz to Hz for correct units on the slope
f_hz = f_thz * 1e12

# YOUR CODE HERE: fit V0 = m*f + b using np.polyfit

# YOUR CODE HERE: compute the proportionality constant and work function
# slope m has units V/Hz = V·s
# since V = J/C, the slope m = h/e, so h = m * e
# the intercept b = -phi/e, so phi = -b * e

print("slope m =", m, "V·s")
print("intercept b =", b, "V")

h_exp = m * e
phi_exp = -b * e

print("proportionality constant =", h_exp, "J·s")
print("work function =", phi_exp, "J")
print("work function =", phi_exp / e, "eV")
```

> SOURCE: code cell from worked version section 5.4. Variable name changed from `h_exp`
> display to "proportionality constant" per the "let them discover it" philosophy.

---

## 4.4 — Cutoff Frequency

`5 points`

> SOURCE: Section 5.4 third code cell from worked version.

```python
# Solve 0 = m*f_c + b for the cutoff frequency
f_c = -b / m
lambda_c = speed_of_light / f_c

print("f_c =", f_c, "Hz")
print("lambda_c =", lambda_c * 1e9, "nm")
```

---

## Problem 5

`10 points`

> SOURCE: Questions drawn from Problem 6 and Problem 7 in the worked version, reduced per TODO.

a) Look up the accepted value of Planck's constant. Compare it to your "proportionality constant." What is this constant?

b) The intercept of your fitted line is negative. Explain why this must be the case, and interpret the intercept in terms of the work function.

c) Using your fitted cutoff frequency $f_c$, explain why increasing intensity cannot produce electron emission when $f < f_c$.

<!-- CLAUDE ADDITION: Question (a) is new — this is the "discovery" moment. Questions (b) and (c)
are from Problem 7b and 7c in the worked version. The original Problems 6 and 7 had 8 questions
total; reduced to 3 here per TODO guidance. -->

---
---

# Final Reflection

> SOURCE: questions from TODO lines 70–73, replacing the original 5-question reflection.

<!-- CLAUDE ADDITION: These are the specific reflection questions from the TODO. They replace
the original 5-question final reflection from the worked/new versions. -->

1. Explain in your own words how the photoelectric effect is evidence of light quantization.

2. Consider how abstraction helps us write reusable programs. In this lab, you performed analysis steps manually, then packaged them into a function. How does organizing code this way help you work with data?

3. What was most difficult about this assignment for you? What would help you improve this skill?

4. *(If you are involved in undergraduate research)* How could you use what you learned in this lab in your research? Give a concrete example.


---
---

# REVISION NOTES

## Source tracking
- **Warmup**: functions primer from worked version + conversion functions from new/scratch
- **Part 1**: experiment intro from worked version + Part 3A plotting from worked version
- **Part 2**: 3B function extraction + 3.2/4.2 apply-to-all from worked/scratch versions
- **Part 3**: ENTIRELY NEW (placeholder, needs intensity data)
- **Part 4**: theory from worked version 5.1–5.3 (condensed) + code from 5.4
- **Final Reflection**: NEW questions from lab3_todo.txt

## Material from originals NOT used (available to pull back in)
- Original Problem 1 questions 1a and 1c (stopping potential = max KE; two wavelengths same V0)
- Section 1.2 "Physical Interpretation of the Data" (from new/scratch) — overview paragraph
- Problem 3 from new version (which wavelengths eject electrons given phi; why freq not wavelength)
- Full theory sections 5.1 (Fermi energy discussion) and 5.2 (cutoff frequency, classical vs quantum detailed argument)
- Problem 5 questions b and d from worked version
- Problem 7 questions a, d, e from worked version
- Original 5-question Final Reflection

## Open items
- [ ] **Part 3 needs intensity-varied data files** — the biggest blocker
- [ ] Figures: TODO says replace Byjus images with custom figures
- [ ] Figures: TODO says add flowchart figures for the function mappings
- [ ] The `1e-3` in `wavelength_nm_to_thz` — added comments but could also restructure to two explicit steps
- [ ] Point values need balancing across the new structure
- [ ] Decide how much of the full theory (Fermi level, classical vs quantum argument) to keep in Part 4
