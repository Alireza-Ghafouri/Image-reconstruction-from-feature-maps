# Image Reconstruction from Feature Maps

## 📌 Overview
This project aims to investigate the feature representations learned by convolutional neural networks (CNNs) by reconstructing images from the feature maps of different layers of a pre-trained **AlexNet** model. The idea is inspired by the paper ["Understanding Deep Image Representations by Inverting Them"](https://arxiv.org/pdf/1506.02753), where the authors attempt to reconstruct images from the feature vectors extracted from different layers of **AlexNet** trained on ImageNet.

## 📂 Dataset
We use the **Tiny ImageNet** dataset instead of the full ImageNet dataset. The dataset consists of 200 classes, out of which 20 classes are randomly selected for training and evaluation.
- **50 images per class** are used for evaluation.
- The remaining images are used for training.
- Training and evaluation sets are **uniformly sampled** from all classes.

## 🏗 Model Structure
We train separate reconstruction models for feature maps extracted from four different layers of **AlexNet**:
1. `conv2`
2. `conv5`
3. `fc6`
4. `fc8`

- The reconstruction architectures are based on the structures provided in **Tables 7 and 8** of the source paper.
- The depth of convolutional reconstruction networks is **reduced to half** to optimize computation and memory usage.
- The **pre-trained weights of AlexNet** are **frozen** during training.

## ⚙️ Installation & Setup
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Project
Simply execute the following command to start training:
```bash
python main.py
```

### 3️⃣ Configuration
All hyperparameters, dataset selection, saving paths, and other settings can be easily modified in the `config.py` file.

## 📊 Outputs
After training, the following results will be available in the specified paths:
- **Loss plots**
- **Visualization of reconstructed images**

## 📜 Reference
- Dosovitskiy, A., & Brox, T. (2016). ["Understanding Deep Image Representations by Inverting Them."](https://arxiv.org/pdf/1506.02753) arXiv preprint arXiv:1506.02753.

## 📌 Notes
- The chosen class IDs will be reported in the experiment logs.
- The pre-trained **AlexNet model** is used from `torchvision.models`.
- The reconstruction loss used is **L2 Loss (Mean Squared Error).**
