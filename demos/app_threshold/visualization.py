import seaborn as sns
from ing_theme_matplotlib import mpl_style
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

mpl_style(dark=False)


def plot_confusion_matrix(probs, actuals, threshold=0.5):
    # Convert predictions to binary labels using the threshold
    binary_predictions = [1 if pred >= threshold else 0 for pred in probs]

    # Compute confusion matrix
    cm = confusion_matrix(actuals, binary_predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap='Blues')
    disp.ax_.set_title('Confusion Matrix')
    disp.ax_.set_xlabel('Predicted Label')
    disp.ax_.set_ylabel('True Label')
    # plt.show()

    return disp.figure_


def plot_probability_distribution(
    probs, actuals, threshold=0.5, plot_title="Predictions for both classes"
):
    """
    Density plots of predicted probabilities vs actuals

    Args:
        probs (np.array): predicted probabilities, float, 0-1
        actuals (np.array): actual, integer, 0 or 1
        threshold (float, optional): [description]. Defaults to 0.5.
        plot_title (str, optional): [description]. Defaults to
            "Predictions for both classes".

    Returns:
        matplotlib.plt: plot object
    """

    # Start plot object
    fig, ax = plt.subplots()
    fig.set_figheight(3)

    # Plot density (using kernel density estimation)
    sns.kdeplot(probs[actuals == 1], shade=True, label="True", ax=ax)
    sns.kdeplot(probs[actuals == 0], shade=True, label="False", ax=ax)

    # Add threshold line with text
    ax.axvline(threshold, 0, 1, color="k", linestyle="dashed", linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    if threshold <= 0.5:
        plt.text(threshold + 0.02, max_ylim * 0.9, f"threshold = {threshold}")
    else:
        plt.text(threshold - 0.24, max_ylim * 0.9, f"threshold = {threshold}")

    # Add plot labels
    ax.set(xlabel="Predicted Probability", ylabel="Density Estimation")
    ax.set(yticklabels=[])
    ax.set_title(plot_title)

    # plot dimensions
    ax.set_xlim(0, 1)
    plt.tight_layout()

    return fig, ax
