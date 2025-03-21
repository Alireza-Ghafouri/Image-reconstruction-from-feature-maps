import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Conv5_Decoder_Net(nn.Module):

    def __init__(self):
        super(Conv5_Decoder_Net, self).__init__()
        self.model_conv = models.alexnet(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # kernel
        self.conv1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv3.weight)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv1.weight)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv2.weight)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv3.weight)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv4.weight)

        self.upconv5 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=3,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv5.weight)

    def forward(self, x):
        x = self.model_conv.features(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv5(x), negative_slope=0.2)

        return x


class Conv2_Decoder_Net(nn.Module):

    def __init__(self):
        super(Conv2_Decoder_Net, self).__init__()
        self.model_conv = models.alexnet(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # kernel
        self.conv1 = nn.Conv2d(
            in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv3.weight)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv1.weight)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv2.weight)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv3.weight)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv4.weight)

        self.upconv5 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=3,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv5.weight)

    def forward(self, x):
        x = self.model_conv.features[:6](x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv5(x), negative_slope=0.2)

        return x


class FC8_Decoder_Net(nn.Module):

    def __init__(self):
        super(FC8_Decoder_Net, self).__init__()
        self.model_conv = models.alexnet(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # kernel
        self.fc1 = nn.Linear(in_features=1000, out_features=4096)
        nn.init.kaiming_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        nn.init.kaiming_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(in_features=4096, out_features=4096)
        nn.init.kaiming_uniform_(self.fc3.weight)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv1.weight)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv2.weight)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv3.weight)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv4.weight)

        self.upconv5 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=3,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv5.weight)

    def forward(self, x):
        x = self.model_conv(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape
        x = F.leaky_relu(self.upconv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv5(x), negative_slope=0.2)

        return x


class FC6_Decoder_Net(nn.Module):

    def __init__(self):
        super(FC6_Decoder_Net, self).__init__()
        self.model_conv = models.alexnet(pretrained=True)
        self.model_conv.classifier = self.model_conv.classifier[:4]
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # kernel
        self.fc1 = nn.Linear(in_features=4096, out_features=4096)
        nn.init.kaiming_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        nn.init.kaiming_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(in_features=4096, out_features=4096)
        nn.init.kaiming_uniform_(self.fc3.weight)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv1.weight)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv2.weight)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv3.weight)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv4.weight)

        self.upconv5 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=3,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        nn.init.kaiming_uniform_(self.upconv5.weight)

    def forward(self, x):
        x = self.model_conv(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape
        x = F.leaky_relu(self.upconv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.upconv5(x), negative_slope=0.2)

        return x
