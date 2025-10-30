def distance_mc_runner(answer_distance, answer_interact, axis_limit=10):
    """
    Validate multiple-choice answers and launch an interactive plot that
    visualizes two points, the segment connecting them, and their Euclidean distance.

    Parameters
    ----------
    answer_distance : {'A','B','C','D'}
        Multiple-choice selection for the distance formula.
        Correct answer is 'C': sqrt((dx)^2 + (dy)^2).

    answer_interact : {'A','B','C','D'}
        Multiple-choice selection for the widgets.interact signature with step=1.
        Correct answer is 'B': v*_ = (-10, 10, 1) for each slider.

    axis_limit : int or float, optional
        Half-range for both axes; plot window is [-axis_limit, axis_limit] in x and y.

    Returns
    -------
    The ipywidgets UI object created by widgets.interact (for notebook environments).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets

    # Map choices to implementations
    distance_forms = {
        'A': lambda v1_x, v1_y, v2_x, v2_y: np.abs(v1_x - v2_x) + np.abs(v1_y - v2_y),           # Manhattan
        'B': lambda v1_x, v1_y, v2_x, v2_y: (v1_x - v2_x)**2 + (v1_y - v2_y)**2,                  # squared distance
        'C': lambda v1_x, v1_y, v2_x, v2_y: np.sqrt((v1_x - v2_x)**2 + (v1_y - v2_y)**2),         # Euclidean (correct)
        'D': lambda v1_x, v1_y, v2_x, v2_y: np.sqrt(v1_x - v2_x) + np.sqrt(v1_y - v2_y),          # invalid generically
    }

    # --- Validate answers ---
    if answer_distance not in distance_forms:
        raise ValueError("ANSWER_DISTANCE must be one of {'A','B','C','D'}.")
    if answer_interact not in {'A','B','C','D'}:
        raise ValueError("ANSWER_INTERACT must be one of {'A','B','C','D'}.")

    if answer_distance != 'C':
        raise AssertionError("Q1 incorrect: The Euclidean distance is option (C).")
    if answer_interact != 'B':
        raise AssertionError("Q2 incorrect: The correct interact signature with step=1 is option (B).")

    # --- Plotting target function ---
    def distance_between_points(v1_x, v1_y, v2_x, v2_y):
        dist = distance_forms[answer_distance](v1_x, v1_y, v2_x, v2_y)

        plt.figure(figsize=(6, 6))
        # Points
        plt.scatter([v1_x], [v1_y], color='tab:blue', label=f"P₁=({v1_x:.1f},{v1_y:.1f})", zorder=3)
        plt.scatter([v2_x], [v2_y], color='tab:orange', label=f"P₂=({v2_x:.1f},{v2_y:.1f})", zorder=3)
        # Connecting segment
        plt.plot([v1_x, v2_x], [v1_y, v2_y], 'k--', lw=1.6, label=f"‖P₁P₂‖ = {dist:.3f}", zorder=2)

        # Axes, grid, labels
        lim = float(axis_limit)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.axhline(0, color='0.75', lw=1)
        plt.axvline(0, color='0.75', lw=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Distance Between Two Points")
        plt.legend(loc='upper left', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.show()

    # --- Build and return the interactive UI ---
    ui = widgets.interact(
        distance_between_points,
        v1_x=(-10, 10, 1),
        v1_y=(-10, 10, 1),
        v2_x=(-10, 10, 1),
        v2_y=(-10, 10, 1),
    )
    return ui
