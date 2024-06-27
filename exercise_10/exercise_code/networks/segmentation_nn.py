"""SegmentationNN"""

import torch
import torch.nn as nn
from torchvision import models


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        INPUT_CHANNELS = 3
        self.device = self.hp.get("device", "cpu")

        ### Encoder ###
        self.collapse = nn.Conv2d(INPUT_CHANNELS, 1, 1)
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.encode = nn.Conv2d(256, 512, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)

        ### Decoder ###
        self.upsample3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up3 = DoubleConv(256, 256)
        self.up2 = DoubleConv(128, 128)
        self.up1 = DoubleConv(64, 64)
        self.decode = nn.Conv2d(64, num_classes, 1)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        # Encode
        x = self.collapse(x)
        enc1 = self.down1(x) # 240
        x = self.maxpool(enc1)
        enc2 = self.down2(x) # 120
        x = self.maxpool(enc2)
        enc3 = self.down3(x) # 60
        x = self.maxpool(enc3)
        x = self.encode(x) # 30

        # Decode
        x = self.upsample3(x) # 60
        x = self.up3(x + enc3)
        x = self.upsample2(x) # 120
        x = self.up2(x + enc2)
        x = self.upsample1(x) # 240
        x = self.up1(x + enc1)
        x = self.decode(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()


if __name__ == "__main__":
    from torchinfo import summary

    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")
