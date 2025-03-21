import os
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp_imgs, out_imgs, titles, model_name, save_path=None):
    """
    Display and optionally save images with titles.

    Parameters:
    - inp_imgs: List of tensors representing input images.
    - out_imgs: List of tensors representing model output images.
    - titles: List of titles corresponding to each image.
    - save_path: Path to save the plot.
    """
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    num_imgs = len(inp_imgs)  # Total images (input-output pairs)
    num_cols = min(num_imgs, 5)  # Limit columns for better visualization
    num_rows = 2  # One row for inputs, one for outputs

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, 6))

    if num_cols == 1:
        axs = np.array(axs).reshape(2, 1)  # Ensure 2D indexing if only 1 column

    for i in range(num_imgs):
        row = 0  # Input images in row 0
        col = i % num_cols  # Arrange images in columns

        img = inp_imgs[i].numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
        img = std * img + mean  # Unnormalize
        img = np.clip(img, 0, 1)  # Ensure valid pixel range

        axs[row, col].imshow(img)
        axs[row, col].set_title(titles[i * 2])  # Input title
        axs[row, col].axis("off")

        # Plot corresponding output in row 1
        row = 1  # Output images in row 1
        img = out_imgs[i].numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
        img = std * img + mean  # Unnormalize
        img = np.clip(img, 0, 1)

        axs[row, col].imshow(img)
        axs[row, col].set_title(titles[i * 2 + 1])  # Output title
        axs[row, col].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        filename = filename = os.path.join(
            save_path, f"{model_name}_evaluation_plot.png"
        )
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    plt.show()
