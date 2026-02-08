#!/usr/bin/env python3
"""
Transform xrd_student_draft.ipynb according to the todo items above line 26 of lab2_todo.txt.

All edits target xrd_student_draft.ipynb ONLY.
"""
import json
import copy
import uuid

INPUT  = "xrd_student_draft.ipynb"
OUTPUT = "xrd_student_draft.ipynb"  # overwrite in place

def new_id():
    return uuid.uuid4().hex[:8]

def md_cell(source_lines):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source_lines
    }

def code_cell(source_lines):
    """Create a code cell (no outputs, no execution count)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": new_id(),
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def clear_code_cell(cell):
    """Strip outputs and execution_count from a code cell."""
    cell["outputs"] = []
    cell["execution_count"] = None
    return cell

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with open(INPUT) as f:
    nb = json.load(f)

cells = nb["cells"]

# ---------------------------------------------------------------------------
# Helper: find cell index by substring match in source
# ---------------------------------------------------------------------------
def find_cell(substring, start=0):
    for i in range(start, len(cells)):
        src = "".join(cells[i]["source"])
        if substring in src:
            return i
    raise ValueError(f"Cell not found: {substring!r}")


# ===========================================================================
# 1. REMOVE USER'S WORKED ANSWER in cell 10 (Question 1 answer)
# ===========================================================================
idx_q1_answer = find_cell("narrowing the slit causes spatial confinement")
cells[idx_q1_answer]["source"] = [
    "*Your answer here (`double click me!`):*\n"
]

# ===========================================================================
# 2. ADD PYTHON FUNCTIONS PRIMER — insert new markdown cell BEFORE Coding Activity 1
# ===========================================================================
idx_ca1 = find_cell("### Coding Activity 1")

functions_primer = md_cell([
    "### A Quick Primer on Python Functions\n",
    "\n",
    "Throughout this lab you will write **functions** — reusable blocks of code that accept inputs, do something, and (usually) return an output. If you have not written a Python function before, here is the pattern:\n",
    "\n",
    "```python\n",
    "def function_name(input_1, input_2):\n",
    "    \"\"\"\n",
    "    A short description of what this function does.\n",
    "    \"\"\"\n",
    "    # do some computation with the inputs\n",
    "    result = input_1 + input_2\n",
    "    return result\n",
    "```\n",
    "\n",
    "Key points:\n",
    "- `def` tells Python you are *defining* a function.\n",
    "- The **parameters** (e.g. `input_1`, `input_2`) are the values you pass in when you *call* the function.\n",
    "- The **return** statement sends a value back to whoever called the function.\n",
    "- Once defined, you call the function by name: `answer = function_name(3, 5)`.\n",
    "- Functions can call other functions — this is how we build complex programs out of small, understandable pieces.\n",
    "\n",
    "You will see a fully worked example of the `beta()` function below. Then you will complete the `single_slit_intensity()` function yourself, following the same pattern.\n",
    "\n",
    "---"
])

cells.insert(idx_ca1, functions_primer)
# After insertion, all subsequent indices shift by 1.

# ===========================================================================
# 3. PART 1A: Provide beta() as EXAMPLE, make single_slit_intensity() a SKELETON
# ===========================================================================
# The code cell right after Coding Activity 1's Part A markdown is the one we need.
idx_ca1_code = find_cell("def beta(theta, a, wavelength):")
cells[idx_ca1_code] = clear_code_cell(cells[idx_ca1_code])
cells[idx_ca1_code]["source"] = [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# Part A: Single-slit model\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "# ===== EXAMPLE FUNCTION (provided for you) =====\n",
    "# Study this function carefully. Notice how:\n",
    "#   - the function name and parameters match the mathematical expression\n",
    "#   - the docstring explains the formula, parameters, and return value\n",
    "#   - np.pi gives the constant pi, and np.sin computes the sine\n",
    "#   - we use \"return\" to send the result back to the caller\n",
    "\n",
    "def beta(theta, a, wavelength):\n",
    "    \"\"\"\n",
    "    Compute the dimensionless phase parameter for single-slit diffraction:\n",
    "\n",
    "        beta(theta; a, lam) = (pi * a / lam) * sin(theta)\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    theta : array_like\n",
    "        Observation angle in radians.\n",
    "    a : float\n",
    "        Slit width.\n",
    "    wavelength : float\n",
    "        Wavelength (same units as a).\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    beta_val : array_like\n",
    "        Dimensionless phase parameter.\n",
    "    \"\"\"\n",
    "    beta_val = np.pi * a / wavelength * np.sin(theta)\n",
    "    return beta_val\n",
    "\n",
    "\n",
    "# ===== YOUR TURN =====\n",
    "# Complete the function below by following the same pattern as beta().\n",
    "# Your function should:\n",
    "#   1. Call beta(theta, a, wavelength) to get beta_val\n",
    "#   2. Compute I = (sin(beta_val) / beta_val)^2\n",
    "#   3. Return I\n",
    "#\n",
    "# Hint: use np.sin() for the sine function.\n",
    "# Note: don't worry about the singularity at beta=0 for now — NumPy\n",
    "#       will give a warning but still produce a usable result.\n",
    "\n",
    "def single_slit_intensity(theta, a, wavelength):\n",
    "    \"\"\"\n",
    "    Single-slit diffraction intensity:\n",
    "\n",
    "        I(theta) = (sin(beta) / beta)^2\n",
    "\n",
    "    with beta defined above. The removable singularity at beta = 0\n",
    "    should be handled so that I(0) = 1.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    theta : array_like\n",
    "        Observation angle in radians.\n",
    "    a : float\n",
    "        Slit width.\n",
    "    wavelength : float\n",
    "        Wavelength.\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    I : array_like\n",
    "        Dimensionless intensity.\n",
    "    \"\"\"\n",
    "    # TODO: implement this function\n",
    "    # Step 1: call beta() to get beta_val\n",
    "    # Step 2: compute I = (np.sin(beta_val) / beta_val) ** 2\n",
    "    # Step 3: return I\n",
    "    pass\n"
]

# ===========================================================================
# 4. PART 1B: Provide the angle grid EXAMPLE here (move from Part 2C)
#    Also add axis labels instruction with angle grid example.
# ===========================================================================
idx_partB_md = find_cell("#### Part B - Visualize the Diffraction Pattern")
cells[idx_partB_md]["source"] = [
    "#### Part B - Visualize the Diffraction Pattern\n",
    "\n",
    "1. Create an angular grid over a small range (for example $ \\theta \\in [-\\pi / 6, \\pi / 6 ]$). An example is given below.\n",
    "2. Plot $I(\\theta)$ for a reasonable choice of $a$ and $\\lambda$. Plots should have a main title as well as axis titles. Use formatted strings such as `r'$\\lambda$'` for math expressions.\n",
    "3. Vary $a$ while holding $\\lambda$ fixed. Generate new plots and observe how the pattern changes.\n",
    "4. Vary $\\lambda$ while holding $a$ fixed. Generate new plots and observe how the pattern changes.\n",
    "\n",
    "**Creating an angular grid:**  \n",
    "Use `np.linspace(start, stop, num_points)` to create an evenly spaced array of angles. For example:\n",
    "```python\n",
    "theta = np.linspace(-np.pi/6, np.pi/6, 500)\n",
    "```\n",
    "This creates 500 evenly spaced values from $-\\pi/6$ to $\\pi/6$. Use `np.pi` for the constant $\\pi$ — do **not** type a decimal approximation like 3.14."
]

# Clear the Part B code cell so students write it themselves
idx_partB_code = find_cell("# Part B: Visualization template")
cells[idx_partB_code] = clear_code_cell(cells[idx_partB_code])
cells[idx_partB_code]["source"] = [
    "\n",
    "# ------------------------------------------------------------\n",
    "# Part B: Visualization\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "# Step 1: Create an angular grid\n",
    "theta = np.linspace(-np.pi/6, np.pi/6, 500)\n",
    "\n",
    "# Step 2: Choose values for slit width (a) and wavelength\n",
    "a = 0.5\n",
    "wavelength = 0.1\n",
    "\n",
    "# Step 3: Call your single_slit_intensity function\n",
    "I = single_slit_intensity(theta, a, wavelength)\n",
    "\n",
    "# Step 4: Plot I vs theta with appropriate labels\n",
    "# TODO: use plt.plot(), plt.title(), plt.xlabel(), plt.ylabel()\n"
]

# ===========================================================================
# 5. PART 1C: Add scaffolding for locating diffraction minima
# ===========================================================================
idx_partC_md = find_cell("#### Part C - Locate Diffraction Minima")
cells[idx_partC_md]["source"] = [
    "#### Part C - Locate Diffraction Minima\n",
    "\n",
    "Using your intensity function:\n",
    "\n",
    "Numerically identify the first few angles at which the intensity goes to (or near) zero.\n",
    "\n",
    "Compare these values to the theoretical condition\n",
    "$$\n",
    "a \\sin \\theta =m \\lambda.\n",
    "$$\n",
    "You do not need a sophisticated root-finding routine; approximate locations are sufficient.\n",
    "\n",
    "**Hint — one approach:**  \n",
    "You already have `theta` (a NumPy array of angles) and `I` (intensity at each angle). One way to find the minima:\n",
    "\n",
    "1. Loop through `I` and check where the intensity is very small (e.g. `< 0.01`).\n",
    "2. Record the *angle* `theta[i]` at those positions.\n",
    "3. Compare $a \\sin(\\theta_{\\text{min}})$ against $m\\lambda$ for $m = \\pm 1, \\pm 2, \\dots$\n",
    "\n",
    "Useful Python patterns:\n",
    "```python\n",
    "# enumerate lets you loop with both index and value\n",
    "for i, I_val in enumerate(I):\n",
    "    if I_val < 0.01:\n",
    "        print(f\"Near-zero at theta = {theta[i]:.4f} rad\")\n",
    "```\n",
    "\n",
    "```python\n",
    "# Computing a * sin(theta) for comparison\n",
    "print(a * np.sin(theta_min))   # should be close to m * wavelength\n",
    "```"
]

# Clear the Part C code cell
idx_partC_code = find_cell("# identify zeros")
cells[idx_partC_code] = clear_code_cell(cells[idx_partC_code])
cells[idx_partC_code]["source"] = [
    "\n",
    "# ------------------------------------------------------------\n",
    "# Part C: Locate diffraction minima\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "# Step 1: Find angles where I is very small\n",
    "# TODO: loop through I and theta to identify near-zero intensities\n",
    "\n",
    "\n",
    "# Step 2: Compare a*sin(theta_min) to m*lambda\n",
    "# TODO: verify the theoretical condition a*sin(theta) = m*lambda\n"
]


# ===========================================================================
# 6. PART 2: Add hints (np.pi, call previously defined functions, overwrite warning)
# ===========================================================================
idx_ca2 = find_cell("### Coding Activity 2")
cells[idx_ca2]["source"] = [
    "### Coding Activity 2\n",
    "`30 points`\n",
    "\n",
    "- P2.1: Write simple Python functions to model single-slit, double-slit, and lattice-based angular interference\n",
    "- P2.3: Construct small modular programs that separate physical models from visualization and analysis\n",
    "\n",
    "In this challenge, you will construct a minimal two-scatterer interference model and verify that discrete angular maxima arise directly from phase matching. This model will serve as a bridge between the optical double-slit experiment and atomic-scale diffraction.\n",
    "\n",
    "Your goal is not to reproduce experimental data, but to translate the phase-difference equations derived in this section into a simple computational model.\n",
    "\n",
    "**Important tips:**\n",
    "- Use `np.pi` for $\\pi$ — do not type a decimal approximation.\n",
    "- In Part B, your `two_scatterer_intensity()` function should **call** your `phase_difference()` function from Part A. Reuse the code you already wrote rather than rewriting the formula.\n",
    "- Be careful **not** to overwrite a function name with a variable! For example, if you have a function called `phase_difference`, do not create a variable called `phase_difference` — use a different name like `phase_dif` or `dphi`.\n",
    "\n",
    "In Part 1 you were given a complete example function (`beta`) and then wrote `single_slit_intensity` yourself. In this section, you will define all three functions from scratch using the provided skeletons. Each skeleton includes the function name, parameters, docstring, and return type — you fill in the body.\n",
    "\n",
    "---\n",
    "\n",
    "#### Part A - Path Difference and Phase Difference\n",
    "\n",
    "Write a function that computes the path difference\n",
    "\n",
    "$$\n",
    "\\Delta(\\theta; d) = d \\sin \\theta .\n",
    "$$\n",
    "\n",
    "Then write a second function that converts this into a phase difference:\n",
    "\n",
    "$$\n",
    "\\Delta\\phi(\\theta; d,\\lambda) = \\frac{2\\pi}{\\lambda} d \\sin \\theta .\n",
    "$$\n",
    "\n",
    "Your functions should accept $\\theta$ in radians and work for both scalar and NumPy array inputs.\n",
    "\n",
    "---\n",
    "\n",
    "**Expected Deliverables:**\n",
    "\n",
    "1. A function computing $\\Delta(\\theta;d)$.\n",
    "2. A function computing $\\Delta\\phi(\\theta;d,\\lambda)$."
]

# PART 2A code cell: provide skeletons with TODO stubs (students write from scratch)
idx_ca2_code = find_cell("def path_difference(theta, d):")
cells[idx_ca2_code] = clear_code_cell(cells[idx_ca2_code])
cells[idx_ca2_code]["source"] = [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# Part A: Path difference and phase difference\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "def path_difference(theta, d):\n",
    "    \"\"\"\n",
    "    Compute the path difference for two scatterers separated by distance d:\n",
    "\n",
    "        delta(theta; d) = d sin(theta)\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    theta : array_like\n",
    "        Observation angle in radians.\n",
    "    d : float\n",
    "        Separation between scatterers (same units as wavelength).\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    delta : array_like\n",
    "        Path difference.\n",
    "    \"\"\"\n",
    "    # TODO: implement (one line)\n",
    "    pass\n",
    "\n",
    "\n",
    "def phase_difference(theta, d, wavelength):\n",
    "    \"\"\"\n",
    "    Convert path difference into a phase difference:\n",
    "\n",
    "        dphi(theta; d, lam) = (2*pi/lam) * d * sin(theta)\n",
    "\n",
    "    Hint: call your path_difference() function!\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    theta : array_like\n",
    "        Observation angle in radians.\n",
    "    d : float\n",
    "        Separation between scatterers.\n",
    "    wavelength : float\n",
    "        Wavelength (same units as d).\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    dphi : array_like\n",
    "        Phase difference.\n",
    "    \"\"\"\n",
    "    # TODO: implement using np.pi and your path_difference function\n",
    "    pass\n"
]

# PART 2B: keep skeleton, ensure it has TODO and the overwrite warning
idx_ca2B_code = find_cell("def two_scatterer_intensity(theta, d, wavelength):")
cells[idx_ca2B_code] = clear_code_cell(cells[idx_ca2B_code])
cells[idx_ca2B_code]["source"] = [
    "\n",
    "# ------------------------------------------------------------\n",
    "# Part B: Minimal two-scatterer intensity model\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "def two_scatterer_intensity(theta, d, wavelength):\n",
    "    \"\"\"\n",
    "    Minimal two-scatterer interference intensity model:\n",
    "\n",
    "        I(theta) = cos^2(dphi(theta) / 2)\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    theta : array_like\n",
    "        Observation angle in radians.\n",
    "    d : float\n",
    "        Separation between scatterers.\n",
    "    wavelength : float\n",
    "        Wavelength.\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    I : array_like\n",
    "        Dimensionless intensity.\n",
    "    \"\"\"\n",
    "    # TODO: implement\n",
    "    # Step 1: call phase_difference(theta, d, wavelength) — store in a variable\n",
    "    #         like phase_dif (NOT phase_difference — that would overwrite the function!)\n",
    "    # Step 2: compute I = np.cos(phase_dif / 2)**2\n",
    "    # Step 3: return I\n",
    "    pass\n"
]

# PART 2C: Remove the angle grid example (it was moved to Part 1B)
# The Part C code cell still has the theta linspace, d, wavelength setup — simplify it
idx_ca2C_code = find_cell("# Part C: Plotting template")
cells[idx_ca2C_code] = clear_code_cell(cells[idx_ca2C_code])
cells[idx_ca2C_code]["source"] = [
    "# ------------------------------------------------------------\n",
    "# Part C: Plotting (students should experiment)\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "# Create an angular grid, choose d and wavelength, then plot.\n",
    "# (Refer to the angle-grid example in Part 1B if needed.)\n",
    "\n",
    "# TODO: create theta array, choose d and wavelength\n",
    "# TODO: call two_scatterer_intensity(theta, d, wavelength)\n",
    "# TODO: plot I vs theta with labels\n"
]

# ===========================================================================
# 7. PART 3: Make students define an interactive plot (passing functions as params)
# ===========================================================================
# Insert a new Coding Activity 3B section after the mirror plane interactive
# First, renumber: Coding Activity 4 -> 3B
idx_ca4 = find_cell("### Coding Activity 4")
cells[idx_ca4]["source"] = [
    "### Coding Activity 3B\n",
    "`20 points`\n",
    "\n",
    "- P2.1: Write simple Python functions to model lattice-based angular interference\n",
    "- C2.3: Interpret diffraction features in terms of periodicity, connecting angular structure to real-space geometry\n",
    "\n",
    "#### Exploring Mirror Planes\n",
    "\n",
    "Run the interactive tool below to explore how changing the Miller indices affects the planes you obtain and the distances between them.\n",
    "\n",
    "As you explore, pay attention to how increasing $h^2 + k^2 + l^2$ changes the density of planes and the spacing. This relationship will become central when we interpret peak positions in powder XRD patterns."
]

# Now we need to add a new coding sub-activity after the mirror plane interactive
# where students define their OWN interactive plot. Insert after cell 34 (mirror code cell).
idx_mirror_code = find_cell("interactive_mirror_plot()")

# Insert a markdown + code cell for the student interactive plot activity
student_interactive_md = md_cell([
    "#### Part 3C — Define Your Own Interactive Plot\n",
    "`10 points`\n",
    "\n",
    "- P2.3: Construct small modular programs that separate physical models from visualization and analysis\n",
    "\n",
    "You have now used several interactive plots built with `ipywidgets`. In this activity, you will define one yourself.\n",
    "\n",
    "**Your task:** Write a function `plot_bragg_angle(d, wavelength, n)` that:\n",
    "1. Computes the Bragg angle $\\theta$ from $2d\\sin\\theta = n\\lambda$ for given $d$, $\\lambda$, and integer $n$.\n",
    "2. Prints the resulting $\\theta$ in degrees.\n",
    "3. Plots a vertical line at that $\\theta$ on an intensity-vs-angle graph (you can reuse your `two_scatterer_intensity` function from Part 2).\n",
    "\n",
    "Then use `widgets.interact` to make `d` and `wavelength` into sliders.\n",
    "\n",
    "**This demonstrates an important programming concept:** you are passing your previously defined functions as building blocks inside a new function. This is how modular programs are built.\n",
    "\n",
    "**Starter code:**\n",
    "```python\n",
    "import ipywidgets as widgets\n",
    "\n",
    "def plot_bragg_angle(d=2.0, wavelength=1.0, n=1):\n",
    "    # Step 1: compute theta from Bragg's law: sin(theta) = n*lam / (2*d)\n",
    "    # Step 2: plot the two_scatterer_intensity pattern\n",
    "    # Step 3: add a vertical line at the Bragg angle using plt.axvline()\n",
    "    pass\n",
    "\n",
    "widgets.interact(plot_bragg_angle,\n",
    "                 d=(0.5, 5.0, 0.1),\n",
    "                 wavelength=(0.1, 3.0, 0.1),\n",
    "                 n=(1, 5, 1))\n",
    "```"
])

student_interactive_code = code_cell([
    "import ipywidgets as widgets\n",
    "\n",
    "# TODO: define plot_bragg_angle(d, wavelength, n)\n",
    "#   - compute theta from Bragg's law: sin(theta) = n*wavelength / (2*d)\n",
    "#     use np.arcsin() to get theta\n",
    "#   - plot two_scatterer_intensity over an angular grid\n",
    "#   - mark the Bragg angle with plt.axvline()\n",
    "#   - add title and axis labels\n",
    "\n",
    "\n",
    "# TODO: use widgets.interact() to make d and wavelength into sliders\n"
])

# Insert after mirror code cell
cells.insert(idx_mirror_code + 1, student_interactive_md)
cells.insert(idx_mirror_code + 2, student_interactive_code)


# ===========================================================================
# 8. Renumber Coding Activity 5 -> Coding Activity 4
# ===========================================================================
idx_ca5 = find_cell("### Coding Activity 5")
cells[idx_ca5]["source"] = [s.replace("Coding Activity 5", "Coding Activity 4") for s in cells[idx_ca5]["source"]]

# ===========================================================================
# 9. PART 4: Add primer on abstraction / generalizing after one example
# ===========================================================================
# Insert before the coding activity 4 (formerly 5)
idx_ca4_new = find_cell("### Coding Activity 4\n")
abstraction_primer = md_cell([
    "### From Concrete Example to Reusable Abstraction\n",
    "\n",
    "A common pattern in scientific programming is:\n",
    "\n",
    "1. **Do something concrete first** — write code that solves one specific case (e.g., analyze the $\\ce{AgB2}$ spectrum step by step).\n",
    "2. **Identify the repeating structure** — notice which parts stay the same and which parts change (e.g., only the filename changes).\n",
    "3. **Wrap it in a function** — replace the changing parts with parameters. Now you can call the function with different inputs.\n",
    "\n",
    "This is called **abstraction**, and it is one of the most important ideas in programming. You did something similar in Part 1 when you wrote `single_slit_intensity()` — you turned a mathematical formula into a reusable function. In this section, you will do the same thing with a full data-analysis workflow.\n",
    "\n",
    "The starter code below walks through the analysis of one spectrum step by step. Your job is to understand each step, then wrap the whole workflow into a single reusable function called `process_spectrum(filename)`.\n",
    "\n",
    "---"
])
cells.insert(idx_ca4_new, abstraction_primer)


# ===========================================================================
# 10. Clean up all code cells (remove outputs, execution counts)
# ===========================================================================
for cell in cells:
    if cell["cell_type"] == "code":
        clear_code_cell(cell)


# ===========================================================================
# Save
# ===========================================================================
nb["cells"] = cells
with open(OUTPUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Wrote {len(cells)} cells to {OUTPUT}")
