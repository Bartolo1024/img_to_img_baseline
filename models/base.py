from torch import nn
import torch.nn.init as init


class SimpleModel(nn.Module):
    def __init__(self, in_channels, n_classes, upscale_factor=4):
        super(SimpleModel, self).__init__()
        self.activation = nn.ReLU()
        self.stump = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.body1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.body2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(32, upscale_factor ** 2 * n_classes, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.activation(self.stump(x))
        x = self.activation(self.body1(x))
        x = self.activation(self.body2(x))
        x = self.activation(self.head(x))
        x = self.activation(self.pixel_shuffle(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.stump.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.body1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.body2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.head.weight)
