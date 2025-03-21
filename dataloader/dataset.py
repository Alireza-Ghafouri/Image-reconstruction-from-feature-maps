from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class TinyImageDataset(Dataset):

    def __init__(self, train_dir, label_dir, layer, transform=None):

        self.train_dir = train_dir
        self.label_dir = label_dir
        self.layer = layer
        self.transform = transform
        # self.dict = {'conv':(192,192),
        #              'fc'  :(128,128)}
        self.dict = {
            "conv2": (416, 416),
            "conv5": (192, 192),
            "fc6": (128, 128),
            "fc8": (128, 128),
        }

    def __len__(self):
        return len(self.train_dir)

    def __getitem__(self, idx):

        train_img = Image.open(self.train_dir[idx]).convert("RGB")
        label_img = Image.open(self.train_dir[idx]).convert("RGB")

        train_transform = transforms.Compose(
            [
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        label_transform = transforms.Compose(
            [
                transforms.Resize(self.dict[self.layer]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if self.transform:
            train_img = train_transform(train_img)
            label_img = label_transform(label_img)

        return train_img, label_img
