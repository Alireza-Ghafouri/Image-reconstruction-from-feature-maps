import requests
import zipfile
import os
import glob
import random
from tqdm import tqdm
import torch
import numpy as np


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_dataset(url, save_path):
    """Download dataset from a given URL if it does not exist locally, with a progress bar."""
    if os.path.exists(save_path):
        print(f"✅ Dataset already downloaded: {save_path}")
        return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as file, tqdm(
            desc="📥 Downloading dataset", total=total_size, unit="B", unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                pbar.update(len(chunk))  # Update tqdm progress bar
        print("✅ Download complete.")
    else:
        print("❌ Failed to download the file.")


def unzip_dataset(zip_path, extract_to):
    """Extract dataset only if it hasn't been extracted already, with progress bar."""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"✅ Dataset already extracted: {extract_to}")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(
            total=len(file_list), desc="📂 Extracting dataset", unit="files"
        ) as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)  # Update tqdm progress bar
    print("✅ Extraction complete.")


def load_classes(wnids_path, num_classes, seed):
    """Load class names from wnids.txt and randomly select a subset."""
    set_seed(seed)  # Ensures consistent class selection

    with open(wnids_path, "r") as f:
        classes = f.read().splitlines()

    print(f"✅ Total classes available: {len(classes)}")

    chosen_classes = random.sample(classes, num_classes)

    print(f"✅ Selected {num_classes} classes: {chosen_classes}")
    return chosen_classes


def get_image_paths(dataset_dir, chosen_classes, train_split):
    """
    Retrieve image paths for selected classes and split them into train/test sets.

    Args:
        dataset_dir (str): Path to the dataset root directory.
        chosen_classes (list): List of selected class names.
        train_split (int): Number of images per class allocated to the test set.

    Returns:
        tuple: (train image paths, test image paths)
    """
    train_paths = []
    x_train_paths, y_train_paths = [], []
    x_test_paths, y_test_paths = [], []

    for cat in chosen_classes:
        train_paths.append(
            os.path.join(
                dataset_dir, "tiny-imagenet-200", "train", cat, "images", "*.JPEG"
            )
        )

    print("🔄 Gathering image paths...")
    for tp in tqdm(train_paths, desc="📂 Processing categories", unit="category"):
        image_list = glob.glob(tp, recursive=True)
        random.shuffle(image_list)
        x_train_paths += image_list[train_split:]
        x_test_paths += image_list[:train_split]

    y_train_paths = x_train_paths.copy()
    y_test_paths = x_test_paths.copy()

    # print(f"✅ Train set: {len(x_train_paths)} images")
    # print(f"✅ Test set: {len(x_test_paths)} images")

    return train_paths, x_train_paths, x_test_paths, y_train_paths, y_test_paths


def select_eval_data(train_paths, x_train_paths, x_test_paths, num_classes, seed):
    """Select evaluation images from a fixed set of classes."""
    set_seed(seed)  # Ensures consistent image selection

    selected_class_paths = random.sample(train_paths, num_classes)

    train_data_eval, test_data_eval = [], []

    for address in selected_class_paths:
        Train_flag = False
        Test_flag = False
        for a in glob.iglob(address, recursive=True):
            if (a in x_train_paths) and not Train_flag:
                train_data_eval.append(a)
                Train_flag = True
            elif (a in x_test_paths) and not Test_flag:
                test_data_eval.append(a)
                Test_flag = True

    return train_data_eval, test_data_eval


def worker_init_fn(worker_id):
    """Ensures different workers have the same seed."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
