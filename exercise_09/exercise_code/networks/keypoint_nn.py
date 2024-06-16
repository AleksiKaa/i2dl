"""Models for facial keypoint detection"""

import torch
import torch.nn as nn


class KeypointModel(nn.Module):
    """Facial keypoint detection model"""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server

        """
        super().__init__()
        self.hparams = hparams

        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################

        input_size = 96
        output_size = 30

        kernel_dim = self.hparams.get("kernel_dim", 3)
        stride = self.hparams.get("stride", 1)
        pool_size = self.hparams.get("pool_size", 2)
        conv1_out = self.hparams["conv1_channels"]
        conv2_out = self.hparams["conv2_channels"]

        dim = input_size  # 96

        conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv1_out, kernel_size=kernel_dim, stride=stride
        )
        dim = (dim - kernel_dim) // stride + 1  # 94

        conv2 = nn.Conv2d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_dim,
            stride=stride,
        )
        dim = ((dim - kernel_dim) // stride + 1) // pool_size  # 46

        relu = nn.ReLU()
        flatten = nn.Flatten()
        pool = nn.MaxPool2d(pool_size, pool_size)
        output = nn.Linear(dim**2 * conv2_out, output_size)

        self.model = nn.Sequential(
            conv1,
            relu,
            conv2,
            relu,
            pool,
            flatten,
            output,
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):

        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""

    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor(
            [
                [
                    0.4685,
                    -0.2319,
                    -0.4253,
                    -0.1953,
                    0.2908,
                    -0.2214,
                    0.5992,
                    -0.2214,
                    -0.2685,
                    -0.2109,
                    -0.5873,
                    -0.1900,
                    0.1967,
                    -0.3827,
                    0.7656,
                    -0.4295,
                    -0.2035,
                    -0.3758,
                    -0.7389,
                    -0.3573,
                    0.0086,
                    0.2333,
                    0.4163,
                    0.6620,
                    -0.3521,
                    0.6985,
                    0.0138,
                    0.6045,
                    0.0190,
                    0.9076,
                ]
            ]
        )

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
