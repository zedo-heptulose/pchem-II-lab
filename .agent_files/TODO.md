# PChem II Lab Notebooks: Comprehensive Revision TODO

**Generated**: December 29, 2025  
**Based on**: Checklist from `pchem_literature_review_and_checklist.md`

---

## Lab Dependencies

| Package | diffraction | magnetism | nmr | pib | photoelectric | quantum_dots | ruby_laser | vib_rot |
|---------|:-----------:|:---------:|:---:|:---:|:-------------:|:------------:|:----------:|:-------:|
| numpy | x | x | x | x | x | x | x | x |
| pandas | x | x | x | x | x | x | x | x |
| matplotlib | x | x | x | x | x | x | x | x |
| ipywidgets | x | x | x | x | | x | x | x |
| scipy | | x | x | x | | | x | x |
| sympy | | x | | | | | | x |
| psi4 | | x | | x | | | | x |
| rdkit | | x | | x | | | | x |
| py3dmol | | x | | x | | | | x |
| fortecubeview | | | | x | | | | |
| scikit-learn | | | | | | x | | |

All dependencies are included in `chm4411l.yml`. Environment file is up to date.

---

## Summary Dashboard

| Notebook | TILT | Cognitive Load | Questions | Technical | Priority |
|----------|------|----------------|-----------|-----------|----------|
| diffraction | 0/3 | 3 issues | Missing pts | 4 critical | HIGH |
| magnetism | 0/3 | 3 issues | Missing pts | 3 critical | HIGH |
| nmr | 0/3 | 3 large funcs | Missing pts | Fragmented imports | MEDIUM |
| particle_in_a_box | 0/3 | Many large funcs | Missing pts | 3 critical | HIGH |
| photoelectric | 0.5/3 | OK | Has pts | 3 critical | MEDIUM |
| quantum_dots | 0.5/3 | 8 large funcs | Missing pts | 4 critical | HIGH |
| ruby_laser | 1/3 | 2 issues | Missing pts | 4 critical | HIGH |
| vibration_rotation | 0/3 | 4 large funcs | Missing pts | Symbol bug | MEDIUM |
| waves_and_optics | ✅ 3/3 | ✅ OK | ✅ Has pts | ✅ Fixed | DONE |

---

## Phase 1: Critical Bug Fixes (Must Fix Before Use)

### diffraction/xrd_student.ipynb

- [ ] **Fix data file paths** - All `'AgB2.xy'` should be `'data/AgB2.xy'`
- [ ] **Fix empty Part 3 cell** (line ~652-656) - Add `from helper import interactive_mirror_plot; interactive_mirror_plot()`
- [ ] **Fix comment/file mismatch** (line ~743) - Comment says `#AuB2` but loads `AgB2.xy`
- [ ] **Investigate import error** - `plot_spectrum` exists in helper.py but import fails
- [ ] **Fix reflection question numbering** (lines ~908-914) - Shows "1, 10000000, 0, 1"
- [ ] **Fix typos**: "descriptioin" → "description", "transtions" → "transitions"

### magnetism/magnetism_student.ipynb

- [ ] **Fix ipywidgets import** - Verify environment has ipywidgets 8.1.2
- [ ] **Define `oqdm` variable** (line ~910) - Uncomment line 900 OR provide clear instruction
- [ ] **Add SMILES for mqdm/pqdm** - meta- and para-quinodimethane SMILES missing
- [ ] **Fix "orbtial" typo** (line ~533) → "orbital"

### particle_in_a_box/pib_student.ipynb

- [ ] **Fix file path** (line ~126) - Change `'mystery_curve.csv'` to `'data/mystery_curve.csv'`
- [ ] **Fix Part 3 import issue** - Add `import os` and `import shutil` at start of Part 3
- [ ] **Fix syntax errors in student templates**:
  - Line ~457: `energy = #calculate energy here!` → `energy = # YOUR CODE HERE`
  - Line ~846: Fix invalid Python syntax `[#look at the orbitals...]`
- [ ] **Fix question numbering** (line ~760) - Questions numbered "1. 2. 4." - missing #3
- [ ] **Move `scipy.optimize.curve_fit` import** to top (currently in Warmup only)

### photoelectric/photoelectric_student.ipynb

- [ ] **Fix function signature mismatch** (Part 3a, line ~533):
  - Change: `process_spectrum(filename)` to `process_spectrum(filename, plot=True)`
  - helper.py calls it with 2 arguments
- [ ] **Fix placeholder filename** (line ~282):
  - Change `"my_placeholder_filename.csv"` to `'data/Ca_200nm.csv'`
- [ ] **Fix column name hints** (lines ~295-296):
  - Change `xcol = "V"` and `ycol = "A"` to `"Voltage (V)"` and `"Current (pA)"`
- [ ] **Fix helper.py bug** (line ~22) - `table.sort_values()` doesn't modify in place

### quantum_dots/quantumdots_student.ipynb

- [ ] **Fix data file paths**:
  - Line ~85: `"freecodedata1.csv"` → `"data/freecodedata1.csv"`
  - Line ~139: `'freecodedata2.csv'` → `'data/freecodedata2.csv'`
- [ ] **Fix `plot_model_predictions()` function** (line ~1102-1124):
  - Add `accuracy` as parameter instead of referencing global variable
- [ ] **Add scikit-learn to environment** - Not in chm4411l.yml, students can't install on HPC

### ruby_laser/ruby_laser_student.ipynb

- [ ] **Fix data file path** (line ~143):
  - Change `'ruby-data-2/tek0001CH1.csv'` → `'data/tek0001CH1.csv'`
  - Change `skiprows=22` → `skiprows=20` (header on line 21)
- [ ] **Fix column name references** throughout:
  - `'TIME (s)'` → `'TIME'`
  - `'CH1 (V)'` → `'CH1'`
- [ ] **Fix `find_lifetime` function path** (lines ~329-331):
  - Change `'folder/tek00{run_number}CH1.csv'` → `f'data/tek00{run_number:02d}CH1.csv'`
- [ ] **Fix typo** (line ~808): `temperature` → `temperatures`
- [ ] **Complete truncated function** (line ~535): `def calc_AE(tau): return 1 / tau`

### vibration-rotation_hcl-dcl/vibration_rotation_student.ipynb

- [ ] **Fix symbol dependency bug** - `x, omega` symbols used in `plot_qho()` before student defines them
- [ ] **Fix typos**: "osciallator" → "oscillator" (3 occurrences), "conterpart" → "counterpart"
- [ ] **Remove unprofessional comment** (line ~823): `#don't forget to floss`

### waves_and_optics ✅ COMPLETE (January 2026)

**Status**: Fully rewritten as `superposition_quantization_student.ipynb` and `superposition_quantization_worked.ipynb`

All original bugs resolved by rewrite:
- [x] **Fix `K1 not defined` error** - Removed; new quantization section uses debugging exercise
- [x] **Fix spherical harmonics code** - Rewritten using scipy.special.sph_harm in helper.py
- [x] **Fix orbital population** - Now uses `spherical_harmonic_plot(l, m)` function

**New file structure:**
- `superposition_quantization_student.ipynb` - Student version
- `superposition_quantization_worked.ipynb` - Instructor version with solutions
- `helper.py` - Contains `standing_animation()`, `ring_wave_animation()`, `spherical_harmonic_plot()`
- `CHANGELOG.md` - Documents all changes from old to new notebooks

**Old files to archive:** `optics_waves_student.ipynb`, `optics_waves_worked.ipynb`, `optics_waves_scratch.ipynb`

**Pending**: Human testing (run notebooks end-to-end in Jupyter)

---

## Phase 2: Structural Improvements (Quality)

### All Notebooks: Add TILT Framework Header

Add to top of each notebook:

```markdown
# [Lab Title]

## Purpose
In this lab you will:
- **Chemistry**: [2-3 chemistry learning objectives]
- **Coding**: [2-3 coding skills]

**Real-world connection**: [Why this matters]

## Estimated Time: [XX-YY minutes]

## Success Criteria
- [ ] [Checkable item 1]
- [ ] [Checkable item 2]
```

### All Notebooks: Add Point Values

- [ ] diffraction - Add point values to all SHORT RESPONSE QUESTIONS
- [ ] magnetism - Add point values to all SHORT RESPONSE QUESTIONS
- [ ] nmr - Add point values to all SHORT RESPONSE QUESTIONS
- [ ] particle_in_a_box - Add point values to all SHORT RESPONSE QUESTIONS
- [ ] quantum_dots - Add point values to all SHORT RESPONSE QUESTIONS
- [ ] ruby_laser - Add point values to all SHORT RESPONSE QUESTIONS
- [ ] vibration_rotation - Add point values to all SHORT RESPONSE QUESTIONS
- [x] waves_and_optics - ✅ Built into new notebooks

(Note: photoelectric already has point values)

### Create Helper Files (Extract Large Functions)

#### nmr/nmr_helper.py (NEW)
Extract:
- `plot_precession()` (~134 lines)
- `plot_fid()` (~125 lines)
- `plot_nmr_visualization()` (~95 lines)

#### particle_in_a_box/pib_helper.py (NEW)
Extract:
- `plot_wave()` (~37 lines)
- `pib_energy_plot()` (~23 lines)
- `xyz_from_smiles()` (~15 lines)
- `show_molecule()` (~9 lines)
- `create_psi4_molecule()` (~13 lines)
- `fit_function()` (~20 lines)
- `get_gaps()` (~9 lines)
- `calculate_r_squared()` (~9 lines)
- `pib_fit_plot()` (~14 lines)

#### magnetism/magnetism_helper.py (NEW)
Extract:
- `zeeman_plot()` (~162 lines)
- `angular_momentum_coupling()` (~139 lines)
- `Arrow3D` class

#### quantum_dots/quantumdots_helper.py (NEW)
Extract:
- `calculate_mo_energy_levels_tight_binding()` (~24 lines)
- `plot_mo_diagram()` (~15 lines)
- `visualize_mo_schematic()` (~56 lines)
- `plot_interactive_band_structure()` (~127 lines)
- `plot_fluorescence_process()` (~215 lines) - WORST OFFENDER
- `plot_feature_importances()` (~8 lines)
- `plot_model_predictions()` (~20 lines)
- `plot_confusion_matrix()` (~18 lines)

#### vibration-rotation_hcl-dcl/vibration_rotation_helper.py (NEW)
Extract:
- `plot_qho()` (~54 lines)
- `plot_rigid_rotor()` (~66 lines)
- `plot_spherical_vector()` (~56 lines)
- `plot_spectrum()` (~19 lines)

### Consolidate Imports

Each notebook should have ONE import cell at the top. Current fragmented imports:

- [ ] **nmr** - 4 separate import locations (lines 19, 340, 602, 845)
- [ ] **particle_in_a_box** - Multiple scattered (lines 21-23, 69, 205-207, 534-540, 401, 810-811, 880)
- [ ] **magnetism** - 3 separate (lines 18-23, 265-270, 804-809)
- [ ] **quantum_dots** - Multiple (lines 21-26, 677-686, 117, 1024-1027)
- [ ] **vibration_rotation** - 4 separate (lines 21-25, 451-453, 714-718, 927)
- [ ] **ruby_laser** - scipy.optimize at line 421 should be at top
- [ ] **photoelectric** - Redundant imports in Part 1 CODE cell (lines 278-279)

### Add Subgoal Labels

Add meaningful labels to worked examples. Template for computational chemistry:

```python
# Subgoal: Define the molecule
smiles = "C=C"
xyz = xyz_from_smiles(smiles)

# Subgoal: Configure the calculation
psi4.set_output_file('output.dat', False)
psi4.set_memory('1 GB')

# Subgoal: Run and extract results
energy, wfn = psi4.energy('SCF/STO-3G', return_wfn=True)

# Subgoal: Visualize and interpret
view = fortecubeview.plot('cubes')
```

For data analysis:

```python
# Subgoal: Load and inspect data
data = pd.read_csv('data/file.csv')

# Subgoal: Process/transform data
data['new_col'] = data['old_col'] * conversion_factor

# Subgoal: Fit model to data
popt, pcov = curve_fit(model_func, x, y)

# Subgoal: Visualize and report
plt.plot(x, y, 'o', label='Data')
```

---

## Phase 3: Content Improvements (Pedagogy)

### Notebooks Flagged as "Too Long"

#### diffraction/xrd_student.ipynb
- [ ] Consider combining "Basics of Diffraction" Parts 1-3 into 2 parts
- [ ] Move Warmup to optional pre-lab
- [ ] Condense markdown explanations - use bullet points
- [ ] Part 2 (Intensity vs Amplitude) and Part 3 (Diffraction) overlap - merge?
- [ ] Target: 60-90 minutes (currently 8 major parts)

### Notebooks Flagged as "Too Easy/Short"

#### nmr/nmr_student.ipynb
Add coding content:
- [ ] Larmor frequency calculation exercise
- [ ] FID-to-spectrum FFT exercise (students write FFT code)
- [ ] Peak width calculation from T2: `FWHM = 1/(π·T2)`
- [ ] Chemical shift calculation: `δ = (ν - ν_TMS) / ν_spectrometer × 10⁶`

### Problem Labs (Need Major Revision)

#### quantum_dots/quantumdots_student.ipynb - "Worst ugly functions, broken ML"
Options for ML section:
- [ ] **Option A**: Remove warmup ML entirely, simplify Part 4
- [ ] **Option B**: Use real quantum dot literature data instead of synthetic
- [ ] **Option C**: Change to regression (predict emission wavelength from size)
- [ ] Current issues: Synthetic data, arbitrary rules, poor model (100 samples, max_depth=3)

### Add References Sections

Missing from:
- [ ] diffraction (has minimal references)
- [ ] nmr
- [ ] particle_in_a_box
- [ ] photoelectric
- [ ] quantum_dots
- [ ] ruby_laser
- [ ] vibration_rotation
- [ ] waves_and_optics

Magnetism has references (lines 1050-1061) - use as template.

---

## Phase 4: Testing & Verification (You Must Do)

### Run All Notebooks End-to-End
- [ ] diffraction/xrd_student.ipynb
- [ ] magnetism/magnetism_student.ipynb
- [ ] nmr/nmr_student.ipynb
- [ ] particle_in_a_box/pib_student.ipynb
- [ ] photoelectric/photoelectric_student.ipynb
- [ ] quantum_dots/quantumdots_student.ipynb
- [ ] ruby_laser/ruby_laser_student.ipynb
- [ ] vibration-rotation_hcl-dcl/vibration_rotation_student.ipynb
- [ ] waves_and_optics/superposition_quantization_student.ipynb

### Environment Testing
- [ ] Test ipywidgets fix on HPC
- [ ] Test scikit-learn installation on HPC
- [ ] Verify spherical harmonics with current library versions
- [ ] Test all Psi4 calculations complete successfully

---

## Quick Reference: File Locations

| Notebook | Student File | Worked File | Helper File |
|----------|--------------|-------------|-------------|
| diffraction | `diffraction/xrd_student.ipynb` | `diffraction/xrd_worked.ipynb` | `diffraction/helper.py` ✓ |
| magnetism | `magnetism/magnetism_student.ipynb` | `magnetism/magnetism_worked.ipynb` | NEED TO CREATE |
| nmr | `nmr/nmr_student.ipynb` | `nmr/nmr_worked.ipynb` | NEED TO CREATE |
| particle_in_a_box | `particle_in_a_box/pib_student.ipynb` | `particle_in_a_box/pib_worked.ipynb` | NEED TO CREATE |
| photoelectric | `photoelectric/photoelectric_student.ipynb` | `photoelectric/photoelectric_worked.ipynb` | `photoelectric/helper.py` ✓ |
| quantum_dots | `quantum_dots/quantumdots_student.ipynb` | `quantum_dots/quantumdots_worked.ipynb` | NEED TO CREATE |
| ruby_laser | `ruby_laser/ruby_laser_student.ipynb` | `ruby_laser/ruby_laser_worked.ipynb` | NEED TO CREATE |
| vibration_rotation | `vibration-rotation_hcl-dcl/vibration_rotation_student.ipynb` | `vibration-rotation_hcl-dcl/vibration_rotation_worked.ipynb` | NEED TO CREATE |
| waves_and_optics | `waves_and_optics/superposition_quantization_student.ipynb` | `waves_and_optics/superposition_quantization_worked.ipynb` | `waves_and_optics/helper.py` ✓ |

---

## Estimated Effort Summary

| Phase | Effort | Who |
|-------|--------|-----|
| Phase 1: Bug Fixes | 4-6 hours | Claude Code |
| Phase 2: Structure | 6-10 hours | Claude Code |
| Phase 3: Content | 4-8 hours | Claude Code + You |
| Phase 4: Testing | 4-6 hours | You |

**Total: ~20-30 hours of revision work**

---

## Notes for Claude Code

When implementing fixes:
1. Always apply changes to BOTH student AND worked versions
2. Test that imports work after extraction to helper files
3. Preserve existing functionality - don't change behavior unless fixing bugs
4. Keep code simple and readable (students are learning)
5. Use standard Python idioms
