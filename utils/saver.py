import ignite.engine
import utils


class NetSaver(object):
    def __init__(self, save_period, session_id, train_args={}):
        self.sess_dir = session_id
        self.save_period = save_period

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_epoch_end)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.store)

    def on_epoch_end(self, engine):
        if engine.state.epoch != 0 and (engine.state.epoch + 1) % self.save_period == 0:
            self.store(engine, engine.state.epoch)

    def store(self, engine, suffix='final'):
        utils.store_weights(engine.state.model, f'out/{self.sess_dir}/weights_{suffix}.h5')
