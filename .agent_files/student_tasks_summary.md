# Student Tasks Summary by Lab

This document lists all tasks students are asked to complete in each lab notebook.

**Legend:**
- [CODE] = Write/complete code
- [QUESTION] = Written response
- [EXPLORE] = Interactive exploration

---

## 1. diffraction/xrd_student.ipynb

### Warmup: Interactive Computing and Visualization
- [CODE] Answer MC question: select correct Euclidean distance formula (A/B/C/D)
- [CODE] Answer MC question: select correct widgets.interact call signature (A/B/C/D)
- [CODE] Run helper function to check answers and launch interactive plot

### Basics of Diffraction - Part 1: Wave Superposition
- [CODE] Implement `waves_and_superposition(f1,p1,f2,p2)` to compute y_1, y_2, and y_superposition
- [CODE] Create a `SuperpositionVisualizer` instance and call its `interactive()` method
- [EXPLORE] Adjust frequency and phase sliders to observe effects on waves and Fourier transform
- [QUESTION] How do the frequencies of the component waves relate to the frequencies observed in the Fourier transform of their superposition? Why are they related like this?
- [QUESTION] Does changing the phase of each wave affect this relationship?
- [QUESTION] How is superposition related to constructive and destructive interference?

### Basics of Diffraction - Part 2: Intensity vs. Amplitude and the Double Slit Experiment
- [EXPLORE] Run `interactive_intensity_plot()` to explore intensity vs. amplitude relationship
- [EXPLORE] Run `interactive_double_slit_plot()` to explore interference as angle changes
- [QUESTION] At what angles and path length differences was the greatest constructive interference observed?
- [QUESTION] How about for destructive interference?
- [QUESTION] Can you relate the path length to the position of a detector and the angle of diffraction?
- [QUESTION] How are these angles and distances related to the wavelength of light passing through the slits?
- [QUESTION] How is the relationship between light intensity and amplitude similar to the relationship between probability density and wavefunction amplitude for an electron?

### Basics of Diffraction - Part 3: Diffraction
- [EXPLORE] Run `interactive_diffraction_plot()` to explore interference from 25 slits
- [QUESTION] Why does diffraction produce much more pronounced peaks compared to two-slit interference?
- [QUESTION] How would the wave superposition plot change if we simulated an infinite number of slits instead of 25?
- [QUESTION] How can we determine the slit spacing for a diffraction grating if it is unknown?

### X-Ray Crystallography - Part 1: The Crystal Lattice
- [EXPLORE] Run `interactive_crystal_plot()` to explore how lattice parameters affect 2D crystal structure
- [CODE] Use `two_d_crystal_lattice()` to create and title all four primitive 2D Bravais lattices
- [QUESTION] How does the translational symmetry of a crystal simplify our calculations?
- [QUESTION] What is the difference between our theoretical crystal model lattice and true real world crystals?

### X-Ray Crystallography - Part 2: Bragg's Law and Diffraction by Reflection
- [EXPLORE] Run `interactive_reflection_plot()` to explore interference of light reflecting off two points
- [EXPLORE] Run `interactive_bragg_plot()` to explore diffraction by scattering off a crystal lattice
- [QUESTION] If we define the y axis as bisecting the angle of reflection, how does the interference of light reflecting off two points depend on their separation along the x-axis and y-axis? Why?
- [QUESTION] In this case, is the variable θ relative to the y-axis or x-axis?
- [QUESTION] What information does diffraction provide about the structure of a crystal?

### X-Ray Crystallography - Part 3: Mirror Planes and HKL Indices
- [EXPLORE] Run `interactive_mirror_plot()` to explore how Miller indices affect planes and distances
- [QUESTION] What type of crystal is this? How can you tell?
- [QUESTION] If our spectrum records of the intensity of light at angles 2θ, what variable from the Bragg equation do these mirror planes relate to?
- [QUESTION] Based on your current understanding, what is X-ray diffraction actually measuring?

### X-Ray Crystallography - Part 4: Characteristic Absences and Interpreting XRD Spectra
- [CODE] Add comments explaining each line of the provided AgB2 processing code
- [CODE] Create `process_spectrum(filename)` function that loads, plots, and classifies spectrum
- [CODE] Use `process_spectrum()` to process AuB2.xy, NbTa.xy, and Po.xy
- [QUESTION] In terms of waves, what does intensity mean?
- [QUESTION] How does the function classify each lattice? Is the result correct?
- [QUESTION] In your current understanding, what is X-ray diffraction measuring?

### Reflection
- [QUESTION] How does the ability to visualize data and mathematical concepts interactively make it easier to examine patterns and learn?
- [QUESTION] How does examining a simplified version of a system help us understand complex phenomena?
- [QUESTION] How does x-ray diffraction help us decipher the structure of crystalline solids?
- [QUESTION] How does processing spectra in Python compare to processing spectra in Excel?

**Total: ~8 CODE tasks, ~25 QUESTIONS, ~10 EXPLORE tasks**

---

## 2. magnetism/magnetism_student.ipynb

### Warmup: Plotting Surfaces
- [CODE] Run example 3D plot, then create your own plot with a different function

### Magnetism in the Bohr Model
- [CODE] Create sympy function for potential energy U(L, B) of Bohr electron in magnetic field
- [CODE] Make 3D plot showing U as function of L and B
- [QUESTION] From the graph you made, describe the relationship between U, L, and B.
- [QUESTION] Why are we able to generalize this equation from Bohr model to quantum mechanical treatment?

### Zeeman Splitting
- [CODE] Create `interactive_zeeman_plot()` function using `widgets.interact()` with sliders for l, ml, B
- [EXPLORE] Explore relationship between vector and orbital representations
- [QUESTION] What is the relationship between the z component of the angular momentum vector and the shape of the wavefunction?
- [QUESTION] How about the xy component of the vector?
- [QUESTION] Why does the magnetic field not cause an energy change for wavefunctions with ml=0 and l≥1?

### Electron Spin and Total Angular Momentum
- [EXPLORE] Use `interactive_coupling_plot()` to explore how J is related to L and S
- [QUESTION] How is the quantum number j related to l and s?
- [QUESTION] How does the z-projection of J, Jz depend on Sz and Lz?
- [QUESTION] If we can place an electron in one of three p orbitals with ms = ±½, how many possible values of Jz are there?

### Probing Organic Diradicals with Psi4
- [CODE] Create three quinodimethane molecules (ortho, meta, para) using SMILES
- [CODE] Display molecules to verify structures
- [CODE] Calculate singlet energies with `psi4.energy()` using `brokensymmetry=True`
- [CODE] Calculate triplet energies with `psi4.energy()`
- [QUESTION] Rank the three molecules by singlet-triplet energy gap. Which have a triplet ground state?
- [QUESTION] Classify each molecule as paramagnetic or diamagnetic.

### Reflection
- [QUESTION] How does the quantization of angular momentum influence electron interactions with magnetic fields?
- [QUESTION] How can we computationally test whether a system is paramagnetic or diamagnetic? Why is this important?

**Total: ~8 CODE tasks, ~12 QUESTIONS, ~2 EXPLORE tasks**

---

## 3. nmr/nmr_student.ipynb

### Warmup - Signal Processing
- [CODE] Plot the raw signal in time domain (signal vs t)
- [CODE] Compute FFT of signal and plot frequency domain spectrum
- [CODE] Apply low-pass filter (set cutoff frequency, zero out high frequencies)
- [CODE] Apply inverse FFT and plot filtered signal in time domain

### Magnetic Energy Splitting
- [CODE] Create function for ΔE(gamma, sigma, B0) that returns energy splitting
- [CODE] Create function that plots energy vs gamma and sigma, accepting B0 as argument
- [CODE] Create interactive plot with slider for B0
- [EXPLORE] Explore how energy splitting depends on B0
- [QUESTION] How does the energy required to promote a nucleus depend on the nucleus and nearby functional groups?

### Magnetic Precession
- [CODE] Use widgets.interact() with plot_precession() function
- [EXPLORE] Explore precession behavior with different angles, field strengths, and time
- [QUESTION] For what angles of α would precession not occur? Does the frequency of precession depend on α?

### Macroscopic Magnetization and the Free Induction Decay
- [CODE] Use widgets.interact() with plot_fid() function
- [EXPLORE] Explore how FID signal relates to magnetization precession
- [QUESTION] How does macroscopic magnetization behave similarly to spin magnetic moment? How is it different?
- [QUESTION] What is the frequency of the induced voltage equal to? How is it related to the magnetic energy splitting?

### T1 and T2 Relaxation
- [CODE] Use widgets.interact() with plot_nmr_visualization() function
- [EXPLORE] Explore effects of shielding, T1, and T2 on NMR spectra
- [QUESTION] Which features of the spectrum do each of these three parameters correspond to?
- [QUESTION] Why does a larger T2 result in sharper peaks?

### Reflection
- [QUESTION] What physical properties does NMR measure, and what information does it provide about a system?
- [QUESTION] How is NMR connected to statistical mechanics? Describe quantum-level and macroscopic principles.

**Total: ~10 CODE tasks, ~8 QUESTIONS, ~4 EXPLORE tasks**

---

## 4. particle_in_a_box/pib_student.ipynb

### Warmup: Curve Fitting
- [CODE] Define a function `f(x, params...) -> y` that fits the mystery curve data
- [EXPLORE] Use `fit_function(my_function, guess_parameters)` to test different functional forms

### Part 1: Solving the Schrodinger Equation
- [EXPLORE] Use `interactive_pib_plot()` to find k values satisfying boundary conditions
- [QUESTION] What values of k gave a wavefunction which satisfies the boundary condition?
- [QUESTION] What relation do these values have to the length? Write a formula relating k to π, L, and n.
- [QUESTION] (Bonus) What is the significance of the functions e^ikx shown in the plot?

### Part 2: Energy Levels
- [CODE] Complete the `pib_energy(n, L_angs)` function to calculate energy levels
- [EXPLORE] Use `pib_energy_plot()` to explore how E depends on n and L
- [QUESTION] How does visualizing the energy levels in three dimensions help make sense of the equation?
- [QUESTION] Based on this plot, predict the shape of the curve when n = L?
- [QUESTION] How does the energy depend on the number of nodes in the wavefunction?

### Computational Chemistry - Part 1: SMILES
- [CODE] Create and display molecules using SMILES: Benzene, Hexane, 2-Butanol, Butadiene
- [QUESTION] Why is it useful to represent molecules using SMILES strings?
- [QUESTION] Give an example of structural information not explicitly defined in a basic SMILES string.

### Computational Chemistry - Part 2: Hartree-Fock Orbital Energies
- [CODE] Specify octatetraene using SMILES
- [CODE] Convert to psi4 molecule using `create_psi4_molecule()`
- [EXPLORE] Run the SCF calculation and plot orbital energies
- [QUESTION] How are the Hartree Fock equations similar to and different from the Schrodinger Equation?
- [QUESTION] Why might there be differences between orbital energies and particle in a box energies?
- [QUESTION] Is there a range for which the orbital energies have a similar shape to PIB energies?

### Computational Chemistry - Part 3: Visualizing Orbitals
- [CODE] Generate .cube files and visualize orbitals with `fortecubeview.plot()`
- [CODE] Create `pi_indices` list containing indices of π orbitals
- [CODE] Add comments explaining the π orbital energy plotting code
- [QUESTION] When else might visualizing orbitals be useful?
- [QUESTION] How do the computed π orbitals compare to the wavefunctions of the particle in a box?
- [QUESTION] Are the π orbital energies well described by a quadratic curve?

### Making an Empirical Model - Part 1: Defining the Problem
- [CODE] Complete `quantum_energy_gap(n_c)`: fill in `n_e`, `homo_index`, `lumo_index`
- [CODE] Complete `pib_energy_gap(n_c)`: fill in `homo_index`, `lumo_index`, `L`, `pib_lumo_e`, `delta_e`
- [CODE] Use `get_gaps(2, 18)` to obtain HOMO-LUMO gaps for alkenes
- [EXPLORE] Plot and compare energy gaps from both models
- [QUESTION] How similar or different are the predictions made by each model?
- [QUESTION] Does it seem that the simplified model might be more valid for a certain range?

### Making an Empirical Model - Part 2: Optimizing the Model
- [CODE] Create `corrected_pib_model(x, A, B)` function
- [CODE] Add comments explaining each argument in the `curve_fit()` call
- [EXPLORE] Examine the fitted model plot and R² value
- [QUESTION] How well does the model fit the quantum data?
- [QUESTION] Does it seem like the model would be a good fit outside the range it was trained on?

### Reflection
- [QUESTION] How do the boundary conditions give rise to discrete energy levels?
- [QUESTION] How are simplified models complementary to more sophisticated models?

**Total: ~14 CODE tasks, ~18 QUESTIONS, ~6 EXPLORE tasks**

---

## 5. photoelectric/photoelectric_student.ipynb

### Warmup: DataFrames and Tabular Data (10 points)
- [CODE] Create a pandas DataFrame from given lists with clear column labels
- [CODE] Add a new column that is 5× one of the numerical columns
- [QUESTION] MCQ1: What does pandas do when computing a scaled column if one entry is NaN?
- [QUESTION] MCQ2: What happens when you multiply a DataFrame column by a number?
- [QUESTION] MCQ3: What type is `df["GPA"] * 5`?
- [QUESTION] Does multiplying GPA by 5 change the number of rows? Columns?

### PART 1 — Identifying the Stopping Potential (20 points)
- [CODE] Load a CSV spectrum file into a DataFrame using `pd.read_csv()`
- [CODE] Plot current vs. voltage with labels, title, and grid
- [QUESTION] List the column names and state the physical quantity/units for each
- [QUESTION] What does a single row correspond to experimentally?
- [QUESTION] Describe photocurrent behavior as voltage increases; identify significance
- [QUESTION] Estimate stopping potential V_s and report K_max in electronvolts
- [QUESTION] MCQ: If light intensity increases but frequency unchanged, V_s is expected to...

### PART 2 — Processing the Spectrum (20 points)
- [CODE] Filter data to isolate the nonzero-current region for fitting
- [CODE] Perform linear regression on current vs. voltage using `np.polyfit`
- [CODE] Plot filtered data with regression line on same axes
- [CODE] Compute x-intercept using `np.roots()` to find stopping potential V_s
- [QUESTION] Why should data points near zero current be excluded from the regression?
- [QUESTION] Report V_s from fit; compare to visual estimate
- [QUESTION] How would V_s change with longer wavelength light?

### PART 3a — Processing One Spectrum (10 points)
- [CODE] Write function `process_spectrum(filename)` that loads, extracts, fits, plots, returns values
- [CODE] Test function on multiple spectrum files
- [QUESTION] Why is returning numerical values preferable to computing them only in a plotting routine?

### PART 3b — Scaling Up the Analysis (10 points)
- [CODE] Import and run `process_all_data`, store and display DataFrame
- [QUESTION] Why automate analysis across all spectra?
- [QUESTION] Describe the trend in stopping potential as wavelength changes
- [QUESTION] How does this trend support or contradict the classical wave description of light?

### PART 4 — Light Energy Versus Frequency (20 points)
- [CODE] Perform linear regression of V_s vs. frequency
- [CODE] Plot data with fitted regression line
- [CODE] Extract slope and intercept; convert slope from eV·s to J·s
- [CODE] Determine work function from intercept
- [QUESTION] Identify the proportionality constant and describe its physical significance
- [QUESTION] Compare calculated value to accepted value; what is percent error?
- [QUESTION] What is the work function? How does it compare to accepted value?
- [QUESTION] Discuss at least two sources of systematic or experimental error

### REFLECTION (10 points)
- [QUESTION] How did pandas and user-defined functions enable reproducible/scalable analysis?
- [QUESTION] How do experimental results contradict the classical wave model of light?

**Total: ~14 CODE tasks, ~18 QUESTIONS, ~0 EXPLORE tasks**

---

## 6. quantum_dots/quantumdots_student.ipynb

### Warmup - Machine Learning with Scikit-Learn
- [CODE] Fill gaps to load CSV, extract features (X, y), visualize with scatter plot
- [CODE] Explain what `c=y` does in the scatter plot (add comment)
- [CODE] Explain why `.values` is used on DataFrame (add comment)
- [CODE] Classify second dataset using LogisticRegression: fit, predict, visualize
- [QUESTION] For which data set did our model give an accurate prediction? Why?
- [QUESTION] What does this tell us about model selection?

### PART 1 - Molecular Orbital Theory
- [CODE] Define a list of atom counts
- [CODE] Initialize empty `mo_energies_list` variable
- [CODE] Call `plot_mo_diagram()` with mo_energies_list and atom_counts
- [CODE] Define `selected_orbitals` list for orbitals 1-4
- [CODE] Call `visualize_mo_schematic()` with n_atoms=4 and selected_orbitals
- [EXPLORE] Observe how MO energy levels change as number of atoms increases
- [QUESTION] How do the molecular orbital energies change as the number of atoms increases? What is the limiting behavior?
- [QUESTION] How do the molecular orbitals differ in structure? Which are bonding/antibonding?

### PART 2 - Band Theory and Semiconductors
- [CODE] Call `interactive_band_structure()` function
- [EXPLORE] Adjust bandgap and valence fill sliders to create insulator, conductor, and semiconductor
- [QUESTION] How do the energy bands compare to orbital energy levels?
- [QUESTION] How does band structure determine conductivity?
- [QUESTION] How does the size of quantum dots control their bandgap?

### PART 3 - Fluorescence
- [CODE] Call `interactive_fluorescence()` function
- [EXPLORE] Adjust energy gap and laser wavelength to observe fluorescence behavior
- [QUESTION] Why is emitted light usually lower energy than absorbed light?
- [QUESTION] Why are a range of values absorbed instead of a single sharp peak?
- [QUESTION] How are the effects that cause lower energy emission and peak broadening related?

### PART 4 - Quantum Yield
- [CODE] Train the RandomForestClassifier model on X_train and y_train
- [CODE] Make prediction on X_test using trained model
- [CODE] Call `plot_model_predictions()` with X_test, y_test, and correct
- [CODE] Call `plot_confusion_matrix()` with cm
- [QUESTION] What properties of a system might cause higher or lower quantum yield?
- [QUESTION] Why do we separate data into train and test sets?
- [QUESTION] What properties does our ML model predict as most important?
- [QUESTION] How well does our model perform based on the confusion matrix?
- [QUESTION] (Bonus) How does a RandomForestClassifier work?

### Reflection
- [QUESTION] How do band theory, semiconductors, fluorescence, and quantum yield explain quantum dot behavior?
- [QUESTION] What are some ways machine learning techniques can give insight into chemical problems?

**Total: ~15 CODE tasks, ~16 QUESTIONS, ~4 EXPLORE tasks**

---

## 7. ruby_laser/ruby_laser_student.ipynb

### PART 1 - Fluorescence lifetime
- [CODE] Load CSV data and plot raw fluorescence decay curve (TIME vs CH1)
- [CODE] Create a new column for normalized channel 1 data plus offset
- [CODE] Create a new column for natural log of normalized data
- [CODE] Plot the log-transformed data vs time
- [CODE] Set time and signal thresholds to filter data to linear region
- [CODE] Assign x (time column) and y (log column) for linear regression
- [CODE] Calculate y_fit and plot data with linear fit overlay
- [QUESTION] Why do we have to add the minimum value before taking the logarithm? Will this affect lifetime values?
- [QUESTION] How can we ensure we are only choosing values in the linear region for an arbitrary run?

### Automating the Process
- [CODE] Complete the `find_lifetime(run_number, plot=False)` function
- [CODE] Fix the folder path in the filename string
- [CODE] Add plotting code inside the `if plot:` block
- [CODE] Call `find_lifetime()` with plotting enabled for run 1
- [CODE] Call `find_lifetime()` for middle runs with plotting
- [CODE] Call `find_lifetime()` for the last run with plotting
- [CODE] Set `max_index`, `temperature_difference`, and `min_temperature`
- [CODE] Plot τ vs T scatter plot
- [CODE] Fit decaying exponential to τ vs T data using `curve_fit`
- [QUESTION] What are the benefits of automatically processing all data with a function? What are the dangers?
- [QUESTION] Did you run into any issues when applying this function to your whole dataset?

### Finding A_E and A_T
- [CODE] Define `calc_AE(tau)` function implementing `A_E = 1/τ`
- [CODE] Choose temperature threshold and calculate average A_E from low-temperature data
- [CODE] Complete `lifetime_model_AE_only(temperature, AE)` function
- [CODE] Plot experimental lifetimes vs model prediction (A_E only)
- [CODE] Define `calc_AT(tau, temperature)` function
- [CODE] Set min/max temperature bounds and calculate average A_T
- [CODE] Complete `lifetime_model_AE_AT(temperature, AE, AT)` function
- [CODE] Plot experimental lifetimes vs A_E+A_T model prediction
- [QUESTION] How does the asymptotic behavior of our A_E-only model compare to experimental data?

### Calculating Nonradiative Relaxation Parameters
- [CODE] Complete `calc_NT(tau, temperature, AE, AT)` function
- [CODE] Set temperature threshold and calculate N(T) values for high-temp data
- [CODE] Calculate `log_NT` and `reciprocal_T` for Arrhenius plot
- [CODE] Assign `E_trs_over_k` and `freqfact` from slope and intercept
- [CODE] Complete `NT_model(temperature, E_trs_over_k, freqfact)`
- [CODE] Complete `final_lifetime_model()` with all pathways
- [CODE] Plot experimental vs final model prediction
- [CODE] Calculate activation energy `E_trs` in cm⁻¹
- [QUESTION] How does the behavior compare now? What might account for remaining discrepancies?
- [QUESTION] Compare this final model to the initial decaying exponential. What are advantages of each?

### Reflection
- [QUESTION] What values were you able to obtain, and what do they physically mean?
- [QUESTION] How did using coarse approximations enable us to gain more information?
- [QUESTION] How did Python make this data processing task possible compared to Excel?

**Total: ~28 CODE tasks, ~10 QUESTIONS, ~0 EXPLORE tasks**

---

## 8. vibration-rotation_hcl-dcl/vibration_rotation_student.ipynb

### Warmup - Computer Algebra
- [EXPLORE] For each SymPy operation example, create at least two additional examples

### Vibrational Motion - Quantum Harmonic Oscillator
- [CODE] Define symbols `x` and `omega` using SymPy
- [CODE] Create 5 wavefunctions: 3 eigenfunctions and 2 non-eigenfunctions
- [CODE] Visualize each wavefunction using `plot_qho()` function
- [QUESTION] Can a wavefunction in a harmonic potential ever have zero energy? Why or why not?
- [QUESTION] Compare Ĥψ(x) and ψ(x). What do you notice when energy is clearly defined vs. when it isn't?
- [QUESTION] What are the energy eigenvalues for the first few eigenfunctions? Write a formula.

### Rotational Motion - Rigid Rotor
- [EXPLORE] Explore spherical coordinates using `interactive_spherical_plot()`
- [EXPLORE] Visualize rigid rotor wavefunctions using `plot_rigid_rotor(J, m)` for various J and m
- [QUESTION] What boundary conditions must the rigid rotor wavefunction satisfy?
- [QUESTION] What do the values m = 0 and m = ±J represent in terms of motion in θ and φ?
- [QUESTION] What are the allowed energy levels as a function of J, m, or both?
- [QUESTION] Write a function for the degeneracy of rigid rotor energy levels.

### Reaction Thermodynamics: OPT + FREQ
- [CODE] Create SMILES strings for ethene, butadiene, and cyclohexene
- [CODE] Optimize geometry of each molecule using Psi4 and compute ΔE_elec^rxn
- [CODE] Perform frequency calculations and compute ΔG^rxn and ΔH^rxn
- [QUESTION] Why is it important to optimize geometry before calculating frequencies?
- [QUESTION] Compare ΔE_elec^rxn and ΔG^rxn. What does the difference tell us?

### IR Spectra: OPT + FREQ
- [CODE] Extract frequencies and IR intensities from psi4.wavefunction objects
- [CODE] Use `plot_spectrum()` to generate simulated IR spectra for each molecule
- [QUESTION] How might the ability to calculate theoretical IR spectra be useful?
- [QUESTION] How accurate are the computational results compared to real IR spectra?

### Reflection
- [QUESTION] How do shapes of potentials and boundary conditions determine valid wavefunctions?
- [QUESTION] How are molecular motions related to molecular energies and thermodynamics?
- [QUESTION] How do the rigid rotor and QHO models apply to real-world chemistry problems?

**Total: ~10 CODE tasks, ~14 QUESTIONS, ~3 EXPLORE tasks**

---

## Summary Statistics

| Lab | CODE | QUESTIONS | EXPLORE | Total Tasks |
|-----|------|-----------|---------|-------------|
| diffraction | 8 | 25 | 10 | 43 |
| magnetism | 8 | 12 | 2 | 22 |
| nmr | 10 | 8 | 4 | 22 |
| particle_in_a_box | 14 | 18 | 6 | 38 |
| photoelectric | 14 | 18 | 0 | 32 |
| quantum_dots | 15 | 16 | 4 | 35 |
| ruby_laser | 28 | 10 | 0 | 38 |
| vibration_rotation | 10 | 14 | 3 | 27 |

**Observations:**
- **diffraction** has the most tasks overall (43) - confirms "too long" assessment
- **ruby_laser** has the most CODE tasks (28) - very coding-heavy
- **nmr** has the fewest QUESTIONS (8) - confirms "too easy/short" assessment
- **photoelectric** and **ruby_laser** have no EXPLORE tasks - more procedural
- **diffraction** has the most EXPLORE tasks (10) - heavily interactive
