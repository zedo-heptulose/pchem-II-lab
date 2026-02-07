# PChem Lab Revision: Task Triage

## Quick Reference

| Task Type | Who/What | Examples |
|-----------|----------|----------|
| Institutional access | You | Get performance data, coordinate with Dr. Clark, HPC decisions |
| Pedagogical judgment | You | Warmup strategy, question quality, difficulty calibration |
| Testing/execution | You | Run notebooks, verify fixes work in your environment |
| Literature review | Claude Web | Teaching programming, comp chem education, question design |
| Writing/brainstorming | Claude Web | Paper drafts, description rewrites, question ideas |
| File operations | Claude Code | Bug fixes, refactoring, formatting, find/replace |
| Data analysis | Claude Code | Feedback statistics, visualizations |

---

## You Must Do These

### Institutional Access Required
- Get lecture performance data (current year + pre-computational years)
- Coordinate with Dr. Clark on flagged questions
- Email standing wave on sphere code creator re: copyright
- Move students to HPC for consistency (if decided)

### Pedagogical Decisions Required
- Decide: remove warmups vs. rework into chemistry content?
- Decide: create intro-to-Python assignment?
- Decide: ASE vs VMD for orbital visualization?
- Decide: keep/remove Bohr model magnetism?
- Flag bad short-response questions
- Final review of all content changes

### Testing Required (Can't Be Delegated)
- Test ipywidgets fix on HPC
- Test scikit-learn environment setup  
- Verify spherical harmonics with current libraries
- Run all notebooks end-to-end after revisions

---

## Claude Web Tasks

### Research (Do First)
- Literature on teaching programming to beginners
- Computational chemistry education best practices
- What makes good formative assessment questions?

### Writing (After You Decide Direction)
- Draft lab descriptions for paper (goal, objectives, assessment)
- Rewrite verbose descriptions to be concise
- Brainstorm improved short-response questions
- Help structure paper sections

### Analysis Support
- Suggest statistical tests for feedback data
- Help interpret feedback patterns

---

## Claude Code Tasks

### Bug Fixes (High Priority)
- `K1 not defined` in waves_and_optics quantization
- Part 3/4 cascade error in photoelectric
- `"` → `'` for filename string in photoelectric
- "Processing Spectrum" typo
- Missing import box in PIB Part 3 worked version
- Add `#` markers in Making an Empirical Model

### Refactoring
- Extract function definitions to `*_helper.py` files
- Remove verbose library explanations
- Standardize code style

### Formatting
- Add title sections to all notebooks
- Add references sections to all notebooks
- Consistent question/section numbering
- Consistent point value labels

### Data Analysis
- Load feedback into pandas, compute statistics
- Generate visualizations with error bars
- ANOVA or other tests (after you specify)

### Content Generation (You Review)
- Python/pandas/psi4 cheat sheets
- Condensed descriptions (you verify accuracy)

---

## Suggested Workflow

**Phase 0 - Groundwork**
- You: Re-read feedback, get institutional data
- Claude Web: Literature review
- Claude Code: File inventory, identify all notebooks

**Phase 1 - Bug Fixes**  
- Claude Code: Fix syntax/typos/imports
- You: Test in your environment

**Phase 2 - Structure**
- You: Decide warmup strategy, flag questions
- Claude Code: Refactor, standardize formatting

**Phase 3 - Content**
- Claude Web: Draft improved content
- You: Review and approve
- Claude Code: Implement changes

**Phase 4 - Paper**
- Claude Code: Feedback analysis, figures
- Claude Web: Writing assistance
- You: Write and finalize

---

## What Claude Code Cannot Do
- Run/execute Jupyter notebooks
- Test psi4, ipywidgets, or other specialized packages
- Access your HPC or Google Drive
- Know what's appropriate difficulty for your students
- Make final pedagogical calls
