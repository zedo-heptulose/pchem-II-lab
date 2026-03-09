New QD outline

Overview of the experiment:
### The Data:
Students will be given spectra for quantum dots with different radii and made of different elements. The spectra should be photoluminescence spectra, intensity vs wavelength (nm), with only a single peak. Spectra files should include metadata of which radius it has and what elements it's made of (CdSe, CdTe, CdS).

See attached script/ guide for generating synthetic data.

### What students are basically doing:
Students will load this data and explore how excitation energy (and/or wavelength) depends on the radius of the quantum dot for the differnt materials. 

Warmup - probably do ML for the warmup.

Parts 1 and 2 are just processing the spectra to get a table with excitation_energy, lambda, r, material. 

In Part 3, students fit models of increasing sophistication to excitation_energy vs r for the different materials and interpret the parameters. (first particle in a (1D or 3D) box, the Brus's model)
Part 3 ends with the students proposing their *own* model and interpreting it, with a challenge to do better than Brus's model for the given data. 

In Part 4, students use basic machine learning w/ scikit-learn to classify quantum dots as being either CdSe, CdTe, or CdS based on their excitation energy and radius. 

This is a good "real" ML task, but it's pretty simple. For Part 3, feel free to change things but it's important to me that the students get to propose their own model. 

---

Chemistry Learning Objectives
1. [Up to you, align with the structure of the lab]
2. [Up to you, align with the structure of the lab]
3. Interpret models as capturing or neglecting physics for a given system. 
4. [Up to you, align with the structure of the lab]

Programming Learning Objectives
1. Use machine learning for classification with Scikit-Learn.
2. [Up to you, but probably don't repeat material from past labs (we can just use it without making it a learning objective)]
3. [Up to you, but probably don't repeat material from past labs (we can just use it without making it a learning objective)]
---

Warmup - Machine Learning 
 have them use a couple different classification models on the same data, maybe?
 Keep this part at a high level, provide scaffolding if you need to. gentle introduction, concepts not technical rigor

Part 1 - Loading one or a few spectra
  Students load a spectrum, plot it, process it to extract a constant.

Part 2 - Processing *all* the spectra
  Students write a function or loop based on their code in part A, to get a DataFrame with all of the spectral data in it.

Part 3 - Modelling Excitation energy vs R
  Part A - Fitting data with PIB
  Part B - Fitting data with Brus' model
  Part C - Write your own function, can you fit the data better?
  SRQs - You might ask students if they were able to fit the data better, what their parameters might physically mean, about overfitting, comparing models, etc. 

Part 4 - Classification With Machine Learning
  Students use Scikit-Learn to classify the elements in the QD given excitation energy and R. 
  Provide scaffolding, as in the warmup. Maybe give them some freedom to explore classifications made by different models and compare which work best for their data.

Reflection
