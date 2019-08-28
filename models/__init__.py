import re
from . import base
from . import unet


def get_model(model, upscale_factor, batch_norms=False):
    m = re.match(r'unet\(([0-9]+),([0-9]+)\)', model)
    if m:
        return unet.Unet(in_channels=int(m.group(1)), n_classes=int(m.group(2)),
                         batch_norms=batch_norms, upscale_factor=upscale_factor)
    m = re.match(r'simple_conv\(([0-9]+),([0-9]+)\)', model)
    if m:
        return base.SimpleModel(upscale_factor=upscale_factor, in_channels=int(m.group(1)),
                                n_classes=int(m.group(2)))
    raise NotImplementedError
