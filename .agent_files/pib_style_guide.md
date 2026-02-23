# PIB Lab Style Guide

Derived from the Waves, Photoelectric, and XRD worked notebooks. Use this as the reference for all PIB lab writing and coding decisions.

---

## Notebook Structure

- **Section order**: Warmup → Part 1 → Part 2 → … → Reflection
- **Header levels**: `##` for Parts, `###` for Coding Activities and Questions, `####` for sub-tasks (A, B, C)
- **Point values**: inline backtick label directly under the `###` header — `` `20 points` ``
- **Learning objectives**: bullet list at the start of each Coding Activity, using `C4.x` / `P4.x` codes
- **Spacer cells**: a single markdown cell containing `<br></br>` between major sections
- **Separator lines**: `---` at the end of question blocks, before the answer placeholder

---

## Writing Style

- **Short descriptions.** Two to four sentences maximum before any code or activity. Students do not read long prose.
- **Bullet lists over paragraphs** for multi-step instructions.
- **Sub-task labels**: use `####` headers for **Part A**, **Part B**, **Part C** within each Coding Activity.
- **Conversational but precise.** The Waves lab tone ("You may be suspicious that this is more physics than chemistry, but…") is appropriate. Avoid overly formal register.
- **Tell students what they will see, not what they should already know.** Frame activities as exploration ("run this and see what happens") rather than tests.
- **No spoilers.** Do not describe the result before students compute it. If we want them to notice something, put it in an SRQ.
- **Math in LaTeX**, always. Raw strings `r'$\sin(\pi x/L)$'` in code labels; `$$...$$` display blocks in markdown.

---

## Scaffolding Pattern

All three labs follow the same progression. Apply it consistently:

1. **Complete worked example** — a full code cell with comments, ready to run.
2. **Annotate the example** — ask students to add a comment on each line explaining what it does (as in XRD W.B and XRD Part 1.C).
3. **Imitate** — a cell with the same structure but blanks to fill in (`# YOUR CODE HERE`).
4. **Extend** — use what was built in a loop, or with different parameters.

Do not jump straight from explanation to blank cells. Always provide a concrete, runnable example first.

---

## Coding Conventions

### Imports
Single consolidated import cell at the top of each notebook. Order: stdlib → numpy/scipy/pandas → matplotlib → helpers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pib_helper import (some_function, another_function)
```

### Functions
- NumPy-style docstrings (Parameters / Returns sections).
- Functions that students are **given** (not asked to write) should have a comment at the top: `# --- Given: [description] ---` and a note that they don't need to modify it.
- Functions students write: provide the signature and docstring, leave the body as `# YOUR CODE HERE`.

### Subgoal labels
Use `# Subgoal: [verb phrase]` comments to chunk code cells — one subgoal per logical step. This is the main in-code organization tool.

```python
# Subgoal: Load and inspect the data
df = pd.read_csv('data/file.csv')

# Subgoal: Filter to the linear region
mask = (df['V'] > -1.0) & (df['V'] < 0.5)
```

### Scaffolded student cells
Commented-out code is the preferred scaffold for optional/student steps:

```python
# Subgoal: Fit your function to the data
# params, _ = curve_fit(my_function, x_data, y_data, p0=[1, 0])
```

Use `# YOUR CODE HERE` for required blanks, `# YOUR COMMENT HERE` when asking for annotations.

---

## Plotting Conventions

Derived from all three labs. Apply uniformly.

```python
plt.plot(x, y, label='description')       # lines (theory, fits, models)
plt.scatter(x, y, label='description')    # discrete data points
plt.xlabel('x label (units)')
plt.ylabel('y label (units)')
plt.title('Descriptive Title')
plt.legend()
plt.show()
```

- **Units always in axis labels**, in parentheses.
- **Raw strings for LaTeX** in labels: `r'$\psi(x)$'`, `r'$E$ (a.u.)'`.
- **Dashed lines** (`'--'`) for reference curves, analytic solutions, or initial guesses. Solid lines for the main result.
- **`'o-'` marker style** (circles connected by lines) for discrete mode/quantum-number data, as in the Waves lab.
- When comparing two things on the same axes, always use `label=` and call `plt.legend()`.
- One `plt.show()` call per figure. Do not reuse axes across cells.

---

## Question Format

### SRQ block structure
```markdown
### Question N
`X points`

- C4.x: [relevant objective]

a) ...

b) ...

c) ...

---
```

Followed immediately by a markdown cell:
```markdown
*Your answer here (`double click me!`):*
```

### Question content
- **Part a)**: Conceptual/observational — what did you see, why does it happen.
- **Part b)**: Comparative or predictive — how does this change if we change a parameter.
- **Part c)**: Connection to the broader context (real systems, other parts of the lab, chemistry).
- Bonus questions are acceptable, labeled `(bonus)` inline.
- Do **not** tell students what they should have observed in the question stem. Ask them to report and explain it.

---

## Helper Function Usage

- Functions longer than ~10 lines that are **not** the pedagogical focus go in `pib_helper.py`.
- Functions that are **given to students but worth reading** (e.g., a plotting utility) are defined inline in the first code cell of the section with the comment `# --- Given: [name] ---`.
- Import only what is used: `from pib_helper import (foo, bar)`.
- Students should always be able to see the source of inline helper functions. Black-box helpers (imported from the file) are for things like optimization loops and molecular geometry, where implementation details distract from the lesson.

---

## Energy / Units Convention (PIB-specific)

- All PIB calculations in **atomic units**: `ħ = 1`, `m_e = 1`.
- Length conversion: `1 Å = 1.8889 a.u.` — define `ANG_TO_AU = 1.8889` at the top of any cell that needs it.
- Analytic reference energy: `E₁(L=1) = π²/2 ≈ 4.9348 a.u.`
- State this in comments, not just in text, so students can check their numbers.

---

## What the Photoelectric Lab Does Especially Well

- **Abstraction progression is explicit**: Part 1 is manual, Part 2 is "now package that as a function," Part 2.B is "now loop over everything." Name this progression when it happens in PIB.
- **Boolean masks for filtering**: shows the pattern clearly and reuses it. If PIB needs data filtering, use the same idiom.
- **Functions call functions**: good model for `pib_energy_gap` calling `pib_energy`.

## What the Waves Lab Does Especially Well

- **Run this and explore** as the first instruction. Give students a working plot immediately, then ask them to modify parameters.
- **`'o-'` plots for discrete mode data**: energy vs. quantum number should use this style.
- **Boundary condition verification in code**: printing `phi[0]` and `phi[-1]` rather than asking students to reason abstractly.

## What the XRD Lab Does Especially Well

- **Annotation as a learning activity**: asking students to comment existing code forces them to articulate understanding. Use this in PIB for any provided helper that students should understand.
- **Interactive widgets before coding**: see the concept visually, then implement it. If a widget is available, lead with it.
