import ignite.engine
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(object):
    def __init__(self, session_id, eval_images=None):
        log_dir = f'./out/{session_id}'
        self.writer = SummaryWriter(log_dir=log_dir)
        self.eval_images = eval_images
        self.epoch_losses = []

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self.on_iteration_end)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_epoch_end)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.completed)

    def on_iteration_end(self, engine):
        self.writer.add_scalar('BatchLoss/train', engine.state.loss,
                               engine.state.iteration)
        self.epoch_losses.append(engine.state.loss)

    def on_epoch_end(self, engine):
        num_batches = len(self.epoch_losses)
        engine.state.epoch_loss = sum(self.epoch_losses) / num_batches
        self.writer.add_scalar('Loss/train', engine.state.epoch_loss,
                               engine.state.iteration)
        self.epoch_losses = []

    def completed(self, engine):
        if self.eval_images:
            return
        images = engine.state.eval_images
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image('images', grid, 0)
        self.writer.add_graph(engine.state.model, images)
        self.writer.close()
