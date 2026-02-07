# PChem II Computational Labs: Literature Review & Revision Guide

## Executive Summary

This document synthesizes research on assignment design, programming pedagogy, and Jupyter notebook best practices to inform the revision of your physical chemistry computational labs. The key frameworks are:

1. **TILT (Transparency in Learning and Teaching)**: Every assignment needs explicit Purpose, Task, and Criteria
2. **Cognitive Load Theory**: Reduce extraneous load through scaffolding, worked examples, and subgoal labels
3. **Purpose-First Programming**: Students care about what code *achieves*, not how it works internally
4. **Active Learning in Notebooks**: Move from "click-run-done" to genuine exploration

---

## Part I: Triaged Reading List

### Tier 1: Read These First (Core Frameworks)

| Resource | Why It Matters | Time |
|----------|---------------|------|
| **TILT Framework Overview** - tilthighered.com | The single most actionable framework for assignment design. Every assignment needs Purpose/Task/Criteria explicitly stated. | 30 min |
| **Teaching and Learning with Jupyter** - jupyter4edu.github.io/jupyter-edu-book/ | The canonical reference. Chapters 3-4 on pedagogical patterns are gold. | 2 hr |
| **Subgoal Labels paper** - Margulieux et al. (2020) "Reducing withdrawal and failure rates..." | Shows how to structure worked examples to reduce cognitive load. Directly applicable to your code walkthroughs. | 1 hr |

### Tier 2: Deep Dives (Read When Planning Specific Revisions)

| Resource | Topic | When to Read |
|----------|-------|--------------|
| **Weiss (2021)** "A Creative Commons Textbook for Teaching Scientific Computing to Chemistry Students" J. Chem. Educ. | Structure for a full computational chemistry curriculum | When planning semester structure |
| **Menke (2020)** "Series of Jupyter Notebooks Using Python for an Analytical Chemistry Course" J. Chem. Educ. | Scaffolding Python for chemistry students with no prior experience | When revising intro material |
| **van Staveren (2022)** "Integrating Python into a Physical Chemistry Lab" J. Chem. Educ. | 5-notebook sequence, survey data on student confidence | When designing lab sequence |
| **Bravenec & Ward (2023)** "Interactive Python Notebooks for Physical Chemistry" J. Chem. Educ. | Colab-based labs, Maxwell-Boltzmann and kinetics modules | For thermodynamics/kinetics content |
| **EPFL Jupyter Guide** - epfl.ch/education/educational-initiatives/jupyter-notebooks-for-education/ | European perspective on notebook pedagogy, especially labs | For lab design patterns |

### Tier 3: Reference Materials (Consult As Needed)

| Resource | Use Case |
|----------|----------|
| **Cognitive Load Theory in Computing Education** - Falkner et al. (2021) ACM TOCE | When you need to justify design decisions with theory |
| **Sweller (2016)** "Cognitive Load Theory and Computer Science Education" | Theoretical background on worked examples |
| **"Purpose-First Programming"** - Xie (2020) dissertation | If you want to radically rethink code tracing exercises |
| **eChem Project** - Norman et al. (2023) J. Chem. Educ. | Jupyter + quantum chemistry notebooks, nbgrader integration |

### Tier 4: Similar Projects in J. Chem. Ed. (Comparables)

These are directly comparable to your project - read for structure, assessment, and lessons learned:

1. **Hughes & Perry (2025)** "Modular Integration of Python Programming in Undergraduate Physical Chemistry Experiments" - Most recent, modular approach, scaffolded syntax introduction

2. **Patel (2025)** "Modernizing Physical Chemistry: Integrating Computational Chemistry, the Finite Well, and Python Data Visualization in the Particle-in-a-Box Experiment" - Directly relevant to your PIB lab

3. **van der Vaart et al. (2025)** "Interactive Application and Visualization of the Variational Method to Aid Conceptual Understanding of Introductory Quantum Mechanics" - Visualization-focused approach

4. **Pelter et al. (2025)** "Computer Simulation of Vinyl Polymerization: Exercises in Critical Thinking Using Jupyter Notebook" - Critical thinking emphasis

---

## Part II: Key Principles Extracted

### From General Pedagogy

**Chickering & Gamson's Seven Principles** (CRLT, UMich):
1. Encourage student-faculty contact
2. Encourage cooperation among students
3. Encourage active learning
4. Give prompt feedback
5. Emphasize time on task
6. Communicate high expectations
7. Respect diverse talents and ways of learning

**TILT Framework** (Winkelmes):
- **Purpose**: What skills/knowledge will students gain? How does this connect to their lives/careers?
- **Task**: Step-by-step what to do (and what to avoid)
- **Criteria**: Rubric + examples of successful work

### From Programming Pedagogy

**Cognitive Load Management**:
- Novices lack mental models → provide scaffolding
- Working memory is limited (~4 chunks) → break down complex procedures
- **Worked examples** > problem-solving for novices
- **Subgoal labels** help students see structure in procedures

**Subgoal Labels** (Margulieux, Morrison, Catrambone):
- Label groups of steps with meaningful names (not "Step 1, Step 2")
- Example: "Initialize variables", "Set up loop structure", "Process each element"
- Reduces dropout/failure rates, especially for struggling students
- Most effective when in BOTH expository text AND worked examples

**Purpose-First Programming** (Xie, 2020):
- Students care about what code achieves, not internal mechanics
- Code tracing creates high cognitive load + low motivation
- Better: Show purpose first, then reveal mechanism as needed

### From Jupyter Notebook Best Practices

**Teaching and Learning with Jupyter** key patterns:

| Pattern | Description | Your Application |
|---------|-------------|------------------|
| **Fill in the blanks** | Students complete partial code | Your `# TODO` sections |
| **Tweak, twiddle, and frob** | Change parameters, observe effects | Interactive widgets |
| **Top-down sequence** | Show working code first, then explain | Flip your current structure? |
| **Coding as translation** | Convert math/chemistry to code | Good for physical chemistry |
| **Now you try** | Same procedure, different data | Transfer practice |

**Common Pitfalls**:
- "Click-run-and-done" shallow engagement
- Students don't understand code they didn't write
- Out-of-order cell execution causes confusion
- Notebooks too long → cognitive overload

**Length Guidelines**:
- Keep notebooks focused on ONE learning objective
- If >30 min to complete, consider splitting
- Use clear section headers with restart points

### From Chemistry Education Literature

**Key findings from J. Chem. Ed. articles**:

1. **Assume no prior programming experience** - Even students who've "taken a CS class" often lack transfer skills

2. **Scaffold Python syntax within chemistry context** - Don't teach Python generically, teach it through chemistry problems

3. **Prioritize readability over efficiency** - Code should be pedagogically clear, not production-quality

4. **Use familiar chemical problems** - PIB, harmonic oscillator, etc. as vehicles for code learning

5. **Include reflection questions** - Ask students to explain what code does and why

6. **Survey students** - Pre/post attitudes about computing are valuable assessment data

---

## Part III: Actionable Revision Checklist

### For Each Lab Assignment

#### A. Before Writing/Revising

- [ ] **Define learning objectives** (both chemistry AND coding)
- [ ] **Identify prerequisites** - What must students already know?
- [ ] **Estimate time to complete** - Target 60-90 min max
- [ ] **Check difficulty relative to other labs** - Is this an outlier?

#### B. Structure Check (TILT Framework)

- [ ] **Purpose section at top includes:**
  - [ ] What chemistry concepts will students understand?
  - [ ] What coding skills will students practice?
  - [ ] How does this connect to research/careers/real world?
  - [ ] How does this build on previous labs?

- [ ] **Task instructions include:**
  - [ ] Step-by-step what to do
  - [ ] Common mistakes to avoid
  - [ ] Estimated time per section
  - [ ] When to ask for help

- [ ] **Criteria for success include:**
  - [ ] What does correct output look like?
  - [ ] Rubric or checklist for evaluation
  - [ ] Example of successful work (if applicable)

#### C. Code Organization

- [ ] **Worked examples have subgoal labels** (not just Step 1, Step 2)
- [ ] **Large function definitions moved to helper files**
- [ ] **Code cells are short** (<15 lines each)
- [ ] **Each code cell has ONE purpose** (don't combine setup + calculation + plotting)
- [ ] **Comments explain WHY, not just WHAT**
- [ ] **Variable names are descriptive and chemistry-relevant**

#### D. Cognitive Load Management

- [ ] **Scaffolding appropriate for novices:**
  - [ ] Fill-in-the-blank for new concepts
  - [ ] Worked examples before practice problems
  - [ ] Subgoal labels on complex procedures
  
- [ ] **Extraneous load reduced:**
  - [ ] Removed irrelevant code/text
  - [ ] Ugly implementation details hidden in helper files
  - [ ] Descriptions are concise (can students actually read them?)
  - [ ] No "walls of text" - use visuals, equations, diagrams

- [ ] **Germane load encouraged:**
  - [ ] Questions ask students to explain, not just execute
  - [ ] "Why" questions after "what" is established
  - [ ] Opportunities to modify and experiment

#### E. Question Quality

- [ ] **Short-response questions:**
  - [ ] Directly connected to code students just wrote
  - [ ] Ask "why" or "what would happen if"
  - [ ] Have clear, assessable answers
  - [ ] Labeled with point values

- [ ] **Reflection questions:**
  - [ ] Synthesize learning across the lab
  - [ ] Connect to broader chemistry concepts
  - [ ] Prompt metacognition ("What was hardest? What would you do differently?")

#### F. Technical Quality

- [ ] **All code cells run without error** (test end-to-end!)
- [ ] **Clear restart points** if students need to start over
- [ ] **No hidden dependencies** on cell execution order
- [ ] **Package imports at top** of notebook
- [ ] **Data files included** and paths work on target environment

#### G. Accessibility & Equity

- [ ] **Instructions don't assume prior CS knowledge**
- [ ] **Errors are informative** (students know what went wrong)
- [ ] **Multiple pathways to success** where possible
- [ ] **Extension activities for advanced students** (optional)

---

## Part IV: Lab-Specific Recommendations

Based on your notes and the literature:

### High Priority Revisions

| Lab | Key Issues | Recommended Changes |
|-----|------------|---------------------|
| **Diffraction** | Too long, verbose | Split into 2 labs OR ruthlessly cut; apply subgoal labels to XRD workflow |
| **NMR** | Too easy/short | Add Larmor frequency calculations, FID→spectrum coding exercise |
| **Quantum Dots** | Ugly code, broken ML | Hide function definitions; replace or remove ML section entirely |

### Structural Patterns to Apply

**For all computational chemistry workflows (Psi4, etc.):**
```
Subgoal: Define the molecule
    - SMILES string → 3D structure
    - Visualize to confirm correct structure

Subgoal: Configure the calculation
    - Set memory limits
    - Choose method and basis set
    - Name output files

Subgoal: Run and extract results
    - Execute calculation
    - Extract energies/properties from wavefunction
    - Handle errors if calculation fails

Subgoal: Visualize and interpret
    - Plot orbitals/surfaces
    - Connect to chemical intuition
    - Clean up temporary files
```

**For data analysis workflows:**
```
Subgoal: Load and inspect data
    - Read file into DataFrame
    - Check dimensions, data types
    - Plot raw data

Subgoal: Process/transform data
    - Unit conversions
    - Baseline corrections
    - Peak finding

Subgoal: Fit model to data
    - Define model function
    - Set initial parameters
    - Run fit, extract parameters

Subgoal: Visualize and report
    - Plot data with fit
    - Calculate derived quantities
    - Report with uncertainties
```

---

## Part V: Assessment & Iteration

### For Your Paper

Metrics to collect/report:
- Pre/post survey on computational confidence (see Menke 2020 for example)
- Completion rates by lab
- Common errors/misconceptions
- Time-on-task estimates
- Student comments from reflections

### For Continuous Improvement

After each semester:
- Which labs had most incomplete submissions?
- Which short-response questions were poorly answered?
- What technical issues occurred?
- What did students say in reflections?

---

## References (Formatted for J. Chem. Ed.)

### Core References

1. Winkelmes, M.A.; Bernacki, M.; Butler, J.; Zochowski, M.; Golanics, J.; Weavil, K.H. A Teaching Intervention That Increases Underserved College Students' Success. *Peer Review* **2016**, *18*(1/2), 31-36.

2. Margulieux, L.E.; Morrison, B.B.; Decker, A. Reducing Withdrawal and Failure Rates in Introductory Programming with Subgoal Labeled Worked Examples. *Int. J. STEM Educ.* **2020**, *7*, 19.

3. Barba, L.A. et al. *Teaching and Learning with Jupyter*; 2019. https://jupyter4edu.github.io/jupyter-edu-book/

4. Sweller, J. Cognitive Load Theory and Computer Science Education. In *Proceedings of the 47th ACM Technical Symposium on Computing Science Education*; ACM: New York, 2016.

### Chemistry Education References

5. Weiss, C.J. A Creative Commons Textbook for Teaching Scientific Computing to Chemistry Students with Python and Jupyter Notebooks. *J. Chem. Educ.* **2021**, *98*(2), 489-494.

6. Menke, E.J. Series of Jupyter Notebooks Using Python for an Analytical Chemistry Course. *J. Chem. Educ.* **2020**, *97*(10), 3899-3903.

7. van Staveren, M. Integrating Python into a Physical Chemistry Lab. *J. Chem. Educ.* **2022**, *99*(7), 2604-2609.

8. Bravenec, A.D.; Ward, K.D. Interactive Python Notebooks for Physical Chemistry. *J. Chem. Educ.* **2023**, *100*(2), 933-940.

9. Hughes, D.J.; Perry, S.C. Modular Integration of Python Programming in Undergraduate Physical Chemistry Experiments. *J. Chem. Educ.* **2025**, *102*(9), 4005-4016.

10. Norman, P. et al. eChem: A Notebook Exploration of Quantum Chemistry. *J. Chem. Educ.* **2023**, *100*(8), 3153-3162.
