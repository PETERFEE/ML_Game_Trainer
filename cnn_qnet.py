# cnn_qnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os # For saving model

class CNN_QNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], output_size: int):
        """
        Initializes a Convolutional Neural Network for pixel-based game states.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input image (channels, height, width).
                                                  E.g., (4, 84, 84) for 4 stacked 84x84 grayscale frames.
            output_size (int): Number of possible actions.
        """
        super().__init__()
        self.input_channels, self.input_height, self.input_width = input_shape
        self.output_size = output_size
        # Store the device the model is on
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convolutional Layers (example architecture, tune as needed for your specific pixel inputs)
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self._linear_input_size = self._get_conv_output_flattened_size(input_shape)

        # Fully Connected (Linear) Layers
        self.fc1 = nn.Linear(self._linear_input_size, 512)
        self.fc2 = nn.Linear(512, self.output_size)

    def _get_conv_output_flattened_size(self, input_shape: tuple[int, int, int]) -> int:
        """
        Helper function to dynamically calculate the input size for the first fully connected layer
        after the convolutional layers. This runs a dummy tensor through the conv layers.
        """
        dummy_input = torch.autograd.Variable(torch.rand(1, *input_shape)).to(self.device) # Move dummy to device
        output_features = self._forward_conv(dummy_input)
        return output_features.view(output_features.size(0), -1).size(1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through convolutional layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        Input x shape: (batch_size, channels, height, width)
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1) # Flatten the output of conv layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
