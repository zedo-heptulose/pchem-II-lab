import numpy as np
import matplotlib.pyplot as plt


def plot_predictions_vs_truth(X_test, y_test, y_pred, model_name="Model",
                              xlabel="Radius (nm)", ylabel="Excitation energy (eV)"):
    """
    Side-by-side scatter plot comparing true labels to model predictions.

    Parameters
    ----------
    X_test : array, shape (n, 2)
        Test features.
    y_test : array
        True labels.
    y_pred : array
        Predicted labels.
    model_name : str
        Name of the model, used in the right subplot title.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for label in np.unique(y_test):
        mask = y_test == label
        ax1.scatter(X_test[mask, 0], X_test[mask, 1], label=label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title("True labels")
    ax1.legend()

    for label in np.unique(y_pred):
        mask = y_pred == label
        ax2.scatter(X_test[mask, 0], X_test[mask, 1], label=label)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"{model_name} predictions")
    ax2.legend()

    plt.tight_layout()
    plt.show()
