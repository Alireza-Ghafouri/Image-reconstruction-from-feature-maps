from .utils import (
    download_dataset,
    unzip_dataset,
    load_classes,
    get_image_paths,
    select_eval_data,
    set_seed,
    worker_init_fn,
)
from .dataset import TinyImageDataset
from torch.utils.data import DataLoader


def create_dataloaders(
    x_train_paths, y_train_paths, x_test_paths, y_test_paths, batch_size, seed
):
    """
    Create DataLoaders for multiple feature extraction layers (conv2, conv5, fc6, fc8).
    """
    set_seed(seed)  # Ensures consistent data loading

    layers = ["conv2", "conv5", "fc6", "fc8"]
    dataloaders = {}
    dataset_sizes = {}

    for layer in layers:
        train_dataset = TinyImageDataset(x_train_paths, y_train_paths, layer, True)
        test_dataset = TinyImageDataset(x_test_paths, y_test_paths, layer, True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )

        dataloaders[layer] = {"train": train_loader, "val": test_loader}
        dataset_sizes[layer] = {"train": len(train_dataset), "val": len(test_dataset)}

    return dataloaders, dataset_sizes


def create_eval_dataloaders(train_data_eval, test_data_eval, batch_size):
    """
    Create DataLoaders for evaluation images.
    """
    layers = ["conv2", "conv5", "fc6", "fc8"]
    eval_dataloaders = {}

    for layer in layers:
        train_eval_dataset = TinyImageDataset(
            train_data_eval, train_data_eval, layer, True
        )
        test_eval_dataset = TinyImageDataset(
            test_data_eval, test_data_eval, layer, True
        )

        train_eval_loader = DataLoader(
            train_eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_eval_loader = DataLoader(
            test_eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        eval_dataloaders[layer] = {"train": train_eval_loader, "val": test_eval_loader}

    return eval_dataloaders


def prepare_tinyimagenet(
    DATASET_DIR,
    ZIP_PATH,
    DOWNLOAD_URL,
    wnids_path,
    num_classes,
    batch_size,
    eval_num_samples,
    seed,
):
    """
    Ensure TinyImageNet dataset is downloaded, extracted, preprocessed, and return DataLoaders.

    Args:
        DATASET_DIR (str): Root dataset directory.
        ZIP_PATH (str): Path to the downloaded ZIP file.
        DOWNLOAD_URL (str): Dataset download link.
        wnids_path (str): Path to wnids.txt file.
        num_classes (int): Number of random classes to use.
        batch_size (int): Batch size for DataLoader.

    Returns:
        dict: Train and validation dataloaders.
        dict: Dataset sizes.
        dict: Evaluation dataloaders.
    """
    # Ensure dataset is downloaded and extracted
    download_dataset(DOWNLOAD_URL, ZIP_PATH)
    unzip_dataset(ZIP_PATH, DATASET_DIR)

    # Load selected classes
    chosen_classes = load_classes(wnids_path, num_classes, seed)

    # Get image paths
    train_paths, x_train_paths, x_test_paths, y_train_paths, y_test_paths = (
        get_image_paths(DATASET_DIR, chosen_classes, eval_num_samples)
    )

    # Create dataloaders for different layers
    dataloaders, dataset_sizes = create_dataloaders(
        x_train_paths, y_train_paths, x_test_paths, y_test_paths, batch_size, seed
    )

    # Select evaluation data
    train_data_eval, test_data_eval = select_eval_data(
        train_paths, x_train_paths, x_test_paths, num_classes, seed
    )

    # Create evaluation dataloaders
    eval_dataloaders = create_eval_dataloaders(
        train_data_eval, test_data_eval, batch_size
    )

    return dataloaders, dataset_sizes, eval_dataloaders
