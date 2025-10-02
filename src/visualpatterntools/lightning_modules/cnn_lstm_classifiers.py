"""Customized LightningModule Classes for Convolutional Neural Network (CNN) and LSTM
(Long Short-Term Memory) Trainable Neural Net Algorithms

References:
    - https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7
    - https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
"""

import logging

import lightning
import torch
import torch.nn
import torchvision.models


module_logger = logging.getLogger(__name__)


class CNN_LSTM(lightning.LightningModule):
    """

    References:
        - https://github.com/kenshohara/video-classification-3d-cnn-pytorch
        - https://arxiv.org/abs/1512.03385 (He2015 - Deep Residual Learning for Image Recognition)
        - https://arxiv.org/abs/1807.06521 (Woo2018 - CBAM: Convolutional Block Attention Module)
    """

    batch_dim_idx: int = 0  # B
    video_frame_dim_idx: int = 1  # F
    frame_channel_dim_idx: int = 2  # C
    frame_height_dim_idx: int = 3  # H
    frame_width_dim_idx: int = 4  # W

    def __init__(
        self,
        num_classes: int,
        resnet_pretrained: torchvision.models.ResNet101_Weights = None,
        resnet_progress: bool = True,
        resnet_kwargs: dict = None,
        lstm_input_size: int = 300,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        resnet_kwargs: dict = resnet_kwargs or dict()

        # ResNet101 is a ResNet implementation with "Bottleneck" and the Layers:
        #   - Layer 1: (64, 3), stride=1
        #   - Layer 2: (128, 4), stride=2
        #   - Layer 3: (256, 23), stride=2
        #   - Layer 4: (512, 3), stride=2
        self.resnet = torchvision.models.resnet101(
            pretrained=resnet_pretrained,
            progress=resnet_progress,
            **resnet_kwargs,
        )

        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.resnet.fc.in_features, lstm_input_size),
        )

        self.lstm = torch.nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
        )

        self.fc1 = torch.nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.fc2 = torch.nn.Linear(lstm_hidden_size // 2, num_classes)

        self.relu_obj = torch.nn.ReLU(inplace=False)

        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Summary

        The `x: torch.Tensor` dimensions should have the meaning:
            - `0`: Batch Index
            - `1`: Video Frame Index
            - `2`: Frame Image Channel Index (Color)
            - `3`: Frame Height Index
            - `4`: Frame Width Index

        Args:
            x (torch.Tensor): Description

        Returns:
            torch.Tensor: Description
        """
        hidden = None
        x_next: torch.Tensor = None

        num_batches: int = x.size(self.batch_dim_idx)
        num_frames: int = x.size(self.video_frame_dim_idx)

        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range():
            with torch.no_grad():
                x_next = self.resnet(x[:, t, :, :, :])

            # Pass latent representation of frame through lstm and update hidden state
            out, hidden = self.lstm(x_next.unsqueeze(0), hidden)

        # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)

        final_hidden_state = hidden[0][-1]

        x_next = self.fc1(final_hidden_state)

        x = self.relu_obj(x_next)

        x = self.fc2(x)

        return x
