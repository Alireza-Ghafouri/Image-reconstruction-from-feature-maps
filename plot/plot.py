import matplotlib.pyplot as plt
import os


def plot_losses(train_losses, test_losses, model_name, save_path=None, filename=None):
    """
    Plot and optionally save the training and testing losses.

    Parameters:
    - train_losses: List of training losses per epoch.
    - test_losses: List of testing losses per epoch.
    - model_name: Name of the model (used for plot title).
    - save_plot: Boolean indicating whether to save the plot (default: False).
    - filename: Filename to save the plot (default: 'loss_plot.png').
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    ax.plot(range(1, len(test_losses) + 1), test_losses, "r", label="Test")
    ax.legend()
    ax.set_title(f"{model_name} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value")
    if save_path:
        plt.savefig(os.path.join(save_path, filename))
    plt.show()
