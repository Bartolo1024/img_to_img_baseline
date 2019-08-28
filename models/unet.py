import torch
from torch import nn


def double_conv(in_c, out_c, kernel_size=3, stride=1, padding=1, batch_norm=True):
    ret = [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
           nn.ReLU()]
    if batch_norm:
        ret.append(nn.BatchNorm2d(out_c))
    ret.extend([nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()])
    if batch_norm:
        ret.append(nn.BatchNorm2d(out_c))
    return nn.Sequential(*ret)


class Unet(nn.Module):
    def __init__(self, n_classes, depth=3, in_channels=3, batch_norms=True, upscale_factor=2):
        super(Unet, self).__init__()
        self.stump = double_conv(in_channels, 64)
        self.max_pool = nn.MaxPool2d(upscale_factor)
        self.down_blocks = [double_conv(64 * 2 ** idx, 64 * 2 ** (idx + 1), batch_norm=batch_norms)
                            for idx in range(depth)]
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=True)
        self.up_blocks = [double_conv(64 * 3 * 2 ** idx, 64 * 2 ** idx, batch_norm=batch_norms)
                          for idx in reversed(range(depth))]
        self.head = nn.Conv2d(64, n_classes, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.stump(x)
        a_outs = [x]
        for idx, block in enumerate(self.down_blocks):
            x = self.max_pool(x)
            x = block(x)
            if idx + 1 == len(self.down_blocks):
                break
            a_outs.append(x)
        for block, pass_trough in zip(self.up_blocks, reversed(a_outs)):
            x = self.upsample(x)
            x = torch.cat((x, pass_trough), dim=1)
            x = block(x)
        x = self.head(x)
        return x
