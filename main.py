from omegaconf import OmegaConf
import os
from dataloader import prepare_tinyimagenet
from model import Conv2_Decoder_Net, Conv5_Decoder_Net, FC6_Decoder_Net, FC8_Decoder_Net
from train import train_and_evaluate, evaluate_model
from plot import plot_losses
import torch.nn as nn
import torch


def main():

    config = OmegaConf.load(os.path.join("config", "config.yaml"))

    DATASET_DIR = config.paths.data.dataset_dir
    ZIP_PATH = config.paths.data.zip_path
    DOWNLOAD_URL = config.paths.data.download_url
    weights_save_path = config.paths.weights_root
    plots_save_path = config.paths.plots_root

    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(plots_save_path, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    device = torch.device(config.training.device)
    dataloaders, dataset_sizes, eval_dataloaders = prepare_tinyimagenet(
        DATASET_DIR=config.paths.data.dataset_dir,
        ZIP_PATH=config.paths.data.zip_path,
        DOWNLOAD_URL=config.paths.data.download_url,
        wnids_path=config.paths.data.wnids_path,
        num_classes=config.training.num_classes,
        batch_size=config.training.batch_size,
        eval_num_samples=config.training.eval_num_samples,
        seed=config.seed,
    )

    # Define model configurations
    conv2_decoder_net = Conv2_Decoder_Net()
    conv5_decoder_net = Conv5_Decoder_Net()
    fc6_decoder_net = FC6_Decoder_Net()
    fc8_decoder_net = FC8_Decoder_Net()

    model_configs = [
        {
            "model": conv2_decoder_net,
            "dataloaders": dataloaders["conv2"],
            "dataset_sizes": dataset_sizes["conv2"],
            "output_size": (3, 416, 416),
            "model_name": "Conv2",
        },
        {
            "model": conv5_decoder_net,
            "dataloaders": dataloaders["conv5"],
            "dataset_sizes": dataset_sizes["conv5"],
            "output_size": (3, 192, 192),
            "model_name": "Conv5",
        },
        {
            "model": fc6_decoder_net,
            "dataloaders": dataloaders["fc6"],
            "dataset_sizes": dataset_sizes["fc6"],
            "output_size": (3, 128, 128),
            "model_name": "FC6",
        },
        {
            "model": fc8_decoder_net,
            "dataloaders": dataloaders["fc8"],
            "dataset_sizes": dataset_sizes["fc8"],
            "output_size": (3, 128, 128),
            "model_name": "FC8",
        },
    ]

    # Train and evaluate each model
    for model_config in model_configs:
        model_ft, train_losses, test_losses = train_and_evaluate(
            model=model_config["model"],
            dataloaders=model_config["dataloaders"],
            dataset_sizes=model_config["dataset_sizes"],
            device=device,
            num_epochs=config.training.num_epochs,
            criterion=nn.MSELoss(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            step_size=config.lr_scheduler.step_size,
            gamma=config.lr_scheduler.gamma,
            model_name=model_config["model_name"],
            save_path=weights_save_path,
        )

        # Plot and save training loss curves
        plot_losses(
            train_losses,
            test_losses,
            model_name=model_config["model_name"],
            save_path=plots_save_path,
            filename=f"{model_config['model_name']}_loss_plot.png",
        )

        # Evaluate and visualize model performance
        evaluate_model(
            model_ft,
            model_config["dataloaders"]["val"],
            model_name=model_config["model_name"],
            num_samples=config.training.visual_num_samples,
            device=device,
            save_path=plots_save_path,
            seed=config.seed,
        )


if __name__ == "__main__":
    main()
