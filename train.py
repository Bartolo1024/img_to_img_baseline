import click
import torch.nn as nn
from torch import optim
import utils
import utils.tensorboard_logger
import utils.saver
import data
import models
import ignite.engine
from ignite.contrib.handlers import ProgressBar


def create_supervised_trainer(model, optimizer, loss_fn):
    model.train()

    def _update(engine, batch):
        optimizer.zero_grad()
        x, y = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        engine.state.loss = loss.item()

    engine = ignite.engine.Engine(_update)

    @engine.on(ignite.engine.Events.STARTED)
    def initialize(engine):
        engine.state.model = model

    @engine.on(ignite.engine.Events.EPOCH_STARTED)
    def train_mode(engine):
        engine.state.model.train()

    def _attach(plugin):
        plugin.attach(engine)

    engine.attach = _attach
    return engine


@utils.timer('training')
@click.command()
@click.option('--epochs', default=100, help='Number of epochs', type=int)
@click.option('-lr', '--learning-rate', default=0.001, type=float)
@click.option('-uf', '--upscale-factor', default=4, type=int)
@click.option('-sv', '--save-period', default=100, type=int)
@click.argument('model')
@utils.session
def main(model, epochs, learning_rate, upscale_factor, save_period, session_id):
    model = models.get_model(model, upscale_factor=upscale_factor, batch_norms=False)
    dataloader = data.get_dataloader(upscale_factor=upscale_factor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    crit = nn.L1Loss()
    trainer = create_supervised_trainer(model, optimizer, crit)
    trainer.attach(ProgressBar(persist=False))
    trainer.attach(utils.tensorboard_logger.TensorboardLogger(session_id=session_id))
    trainer.attach(utils.saver.NetSaver(save_period, session_id))
    _ = trainer.run(dataloader, max_epochs=epochs)


if __name__ == '__main__':
    main()
