import click
import torch.nn as nn
import PIL.Image
import matplotlib.pyplot as plt
import utils
import utils.saver
import data
import models
import ignite.engine
from ignite.contrib.handlers import ProgressBar


def is_img(path):
    return path.split('.')[-1] in ('png', 'jpg', 'jpeg')


def load_img(filepath, upscale_factor=4, downscale_steps=3):
    img = PIL.Image.open(filepath).convert('YCbCr')
    w, h = img.size
    w = w - (w % upscale_factor ** downscale_steps)
    h = h - (h % upscale_factor ** downscale_steps)
    img = img.resize((w, h))
    y, _, _ = img.split()
    return y


def create_supervised_validator(model, loss_fn, metrics):
    model.eval()

    def _update(engine, batch):
        model.train()
        x, y = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        engine.state.loss = loss.item()

    engine = ignite.engine.Engine(_update)

    def _attach(plugin):
        plugin.attach(engine)

    engine.attach = _attach
    return engine


def validate_on_dataset(dataset_path, model, upscale_factor):
    dataloader = data.get_dataloader(upscale_factor=upscale_factor)
    crit = nn.L1Loss()
    validator = create_supervised_validator(model, crit, {})
    validator.attach(ProgressBar(persist=False))
    _ = validator.run(dataloader, max_epochs=1)


def validate_on_single_img(model, img_path, upscale_factor):
    img = load_img(img_path, upscale_factor)
    w, h = img.size
    size = h // upscale_factor, w // upscale_factor
    x = data.get_input_tranforms(size=size)(img).unsqueeze(0)
    downscaled = data.denormalize(x[0, 0].detach().numpy())
    out_img = model(x).clamp(0)
    out_img = out_img.detach().numpy()
    out_img = data.denormalize(out_img)
    out_img = PIL.Image.fromarray(out_img[0, 0], 'L')
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[1].imshow(downscaled, cmap='gray', vmin=0, vmax=255)
    axs[2].imshow(out_img, cmap='gray', vmin=0, vmax=255)
    plt.show()


@click.command()
@click.option('-w', '--weights', type=str)
@click.option('-uf', '--upscale-factor', default=4, type=int)
@click.argument('model')
@click.argument('data-path')
def main(model, weights, upscale_factor, data_path):
    model = models.get_model(model, upscale_factor, 1, 1, batch_norms=False)
    utils.load_weights(model, weights)
    if is_img(data_path):
        validate_on_single_img(model, data_path, upscale_factor)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
