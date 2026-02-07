# PChem Lab Revision Guide (Claude Code Reference)

## Quick Reference: What Makes a Good Lab

### TILT Framework (Every Lab Needs)
1. **Purpose**: What will students learn (chemistry + coding)? Why does it matter?
2. **Task**: Step-by-step what to do. Estimate time. Flag common mistakes.
3. **Criteria**: Point values on all questions. What does a good answer look like?

---

## Pedagogical Patterns by Lab

| Lab | Primary Patterns | Independence |
|-----|------------------|--------------|
| 1. Waves/Optics | Shift-Enter, Tweak & Twiddle, one Fill-in-Blank | Very Low |
| 2. Diffraction | Worked Example → Your Turn, Target Practice | Low |
| 3. Photoelectric | Target Practice, Modify & Extend | Low-Medium |
| 4. PIB | Worked Example → Your Turn, Target Practice + subgoals | Medium |
| 5. Rot/Vib | Modify & Extend, introduce assertions | Medium |
| 6. Magnetism | Target Practice (students BUILD the visualization) | Medium |
| 7. NMR | Target Practice, Solve from Scratch (with hints) | Medium-High |
| 8. Quantum Dots | Solve from Scratch (hints), Modify & Extend | Medium-High |
| 9. Ruby Laser | Solve from Scratch (checkpoints + assertions) | High |

### Pattern Definitions
- **Shift-Enter**: Run provided code, observe output
- **Fill-in-the-Blank**: Complete gaps in provided code
- **Tweak & Twiddle**: Modify parameters, observe effects
- **Target Practice**: Given goal + template, write code to achieve it
- **Worked Example → Your Turn**: Study example, write analogous code
- **Modify & Extend**: Take working code, add functionality
- **Solve from Scratch**: Given specs only, write complete solution

---

## Subgoal Labels (Use in Code Comments)

### For Psi4 Workflows
```python
# ============================================
# SUBGOAL: Define the molecule
# ============================================

# ============================================
# SUBGOAL: Configure the calculation
# ============================================

# ============================================
# SUBGOAL: Run and extract results
# ============================================

# ============================================
# SUBGOAL: Visualize and interpret
# ============================================
```

### For Data Analysis
```python
# ============================================
# SUBGOAL: Load and inspect data
# ============================================

# ============================================
# SUBGOAL: Process/transform data
# ============================================

# ============================================
# SUBGOAL: Fit model to data
# ============================================

# ============================================
# SUBGOAL: Visualize and report
# ============================================
```

---

## Short Response Question Guidelines

### Question Quality Checklist
- [ ] Assesses a stated learning objective
- [ ] One clear interpretation of what's being asked
- [ ] Requires thinking, not just recall (Apply level or higher)
- [ ] References work student just completed
- [ ] Point value included
- [ ] One question per question (not three disguised as one)

### Bloom's Level Quick Reference
| Level | Use For | Example Stems |
|-------|---------|---------------|
| Remember | Checking prerequisites only | What is...? Define... |
| Understand | After explanations | Explain in your own words... |
| **Apply** | After worked examples | Using your results, calculate... |
| **Analyze** | Connecting concepts | Compare... How does X affect Y... |
| **Evaluate** | End of sections | Which is better and why... |
| Create | Reflection/synthesis | What would you change... |

**Target: Most questions at Apply or higher.**

### CER Framework (for interpretation questions)
Ask students to provide:
- **Claim**: Direct answer (1 sentence)
- **Evidence**: Specific data/results supporting the claim
- **Reasoning**: Scientific principle explaining why evidence supports claim

### Common Problems to Fix
| Problem | Fix |
|---------|-----|
| Too vague ("What do you notice?") | Specify what to focus on and answer format |
| Recall-only / Googleable | Ask about relationships and reasoning |
| Answer embedded in question | Let students evaluate, don't lead |
| Yes/No without justification | Require explanation |
| Disconnected from activity | Reference specific work just completed |
| Multiple questions in one | Split into separate questions |

---

## Code Cell Guidelines

- **Max 15 lines per cell** (one purpose each)
- **Imports at top** of notebook
- **Large functions** → move to helper file, import
- **Every code task** needs: clear instruction, expected output format, how to verify success
- **Add assertions** in Labs 5+ as checkpoints:
```python
assert abs(result - expected) < tolerance, "Check your calculation"
```

---

## Lab-Specific Notes

### Labs Needing Major Revision
- **Diffraction**: Cut from 43 to ~30 tasks. Reduce questions from 25 to ~15.
- **NMR**: Expand significantly. Add equation→code tasks (Larmor freq, chemical shift). Students should code more.
- **Magnetism**: Fix "notebook as app" middle section. Students should BUILD the interactive plot.
- **Quantum Dots**: Move ugly functions to helper file. Rework ML section with clear dataset/task. Add Psi4 excited states.
- **Ruby Laser**: Add checkpoints with assertions. Include troubleshooting guide.

### Skill Progression to Maintain
| Skill | Introduced | Reinforced | Independent |
|-------|------------|------------|-------------|
| Writing functions | Lab 2 | Labs 3-8 | Lab 9 |
| Assertions/tests | Lab 5 | Labs 6-8 | Lab 9 |
| Open-ended tasks | Lab 7 | Lab 8 | Lab 9 |

---

## Revision Workflow

For each lab:
1. Check TILT components (Purpose/Task/Criteria) exist
2. Verify pedagogical pattern matches lab number (see table above)
3. Add subgoal labels to multi-step code workflows
4. Review each question against checklist
5. Ensure code cells are short, commented, with clear instructions
6. Test notebook runs end-to-end without errors
7. Move final to `/mnt/user-data/outputs/`
