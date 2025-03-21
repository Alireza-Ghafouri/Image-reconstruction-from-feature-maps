import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import numpy as np
import time
import copy
from tqdm import tqdm
from plot import imshow
import os
from .utils import set_seed


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    dataset_sizes,
    type,
    device,
    save_path,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    train_epoch_losses = []
    test_epoch_losses = []
    epochs_lr_list = []

    print("\n--->", type, " training: \n")
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == "train":
                scheduler.step()
                print("LR:{}".format(optimizer.param_groups[0]["lr"]))

            epoch_loss = running_loss / dataset_sizes[phase]

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "train":
                train_epoch_losses.append(epoch_loss)
            elif phase == "val":
                test_epoch_losses.append(epoch_loss)

            epochs_lr_list.append(optimizer.param_groups[0]["lr"])

            # deep copy the model
            if phase == "val" and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            os.path.join(save_path, "{}_Last_Model.pt".format(type)),
        )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_epoch_losses, test_epoch_losses


def train_and_evaluate(
    model,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs,
    criterion,
    lr,
    weight_decay,
    step_size,
    gamma,
    model_name,
    save_path,
):
    """
    Train and evaluate the given model.

    Parameters:
    - model: The neural network model to train.
    - dataloaders: Dictionary containing 'train' and 'test' DataLoader objects.
    - dataset_sizes: Dictionary containing the sizes of the 'train' and 'test' datasets.
    - num_epochs: Number of epochs to train the model.
    - device: Device to run the model on ('cpu' or 'cuda').
    - criterion: Loss function (default: nn.MSELoss()).
    - lr: Learning rate for the optimizer (default: 0.001).
    - step_size: Period of learning rate decay (default: 5).
    - gamma: Multiplicative factor of learning rate decay (default: 0.1).
    - weight_decay: Weight decay (L2 penalty) (default: 0.0001).
    - model_name: Name of the model (used for labeling plots and saving files).

    Returns:
    - model_ft: Trained model.
    - train_epoch_losses: List of training losses per epoch.
    - test_epoch_losses: List of testing losses per epoch.
    """
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_sched = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.to(device)
    model_ft, train_epoch_losses, test_epoch_losses = train_model(
        model,
        criterion,
        optimizer,
        lr_sched,
        num_epochs=num_epochs,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        type=model_name,
        device=device,
        save_path=save_path,
    )
    return model_ft, train_epoch_losses, test_epoch_losses


def evaluate_model(model, dataloader, model_name, num_samples, device, save_path, seed):
    """
    Evaluates a model on a dataset and visualizes input-output pairs.

    Parameters:
    - model: The trained model to evaluate.
    - dataloader: The dataloader for the dataset (e.g., test or validation set).
    - model_name: Name of the model (used for visualization).
    - num_samples: Number of sample images to visualize.
    - device: Device to run the model on ("cpu" or "cuda").

    Returns:
    - inp_imgs: List of input images.
    - out_imgs: List of model-generated output images.
    """
    set_seed(seed)  # Ensures same images for all models
    model.to(device)
    model.eval()

    inp_imgs, out_imgs = [], []
    titles = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(
            dataloader
        ):  # Assuming dataset returns (image, label)
            if i >= num_samples:
                break

            inputs = inputs.to(device)
            outputs = model(inputs).detach().cpu()

            for j in range(inputs.shape[0]):  # Loop over batch
                if len(inp_imgs) >= num_samples:  # Ensure we don't exceed num_samples
                    break

                inp_imgs.append(inputs[j].cpu())  # Individual images
                out_imgs.append(outputs[j])  # Individual model outputs

                titles.append(f"Input Image {len(inp_imgs)}")
                titles.append(f"{model_name} Output {len(out_imgs)}")

    # Visualize results
    imshow(inp_imgs, out_imgs, titles, model_name, save_path=save_path)

    return inp_imgs, out_imgs
