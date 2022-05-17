import tensorflow as tf

class LossWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps=20000):
        super().__init__()
        self.warmup_steps = warmup_steps

    def on_train_begin(self, logs):
        self.step = 0

        self.max_l_t = float(self.model.texture_swap_loss_weight)
        self.max_l_s = float(self.model.structure_swap_loss_weight)
        self.max_l_w = float(self.model.warp_loss_weight)

    def on_batch_begin(self, batch, logs):
        if self.step > self.warmup_steps:
            return
        T = float(self.warmup_steps)
        t = float(self.step)
        l_t = self.max_l_t  * (t / T)
        l_s = self.max_l_s * (t / T)
        l_w = self.max_l_w * (t / T)
        #l_c = self.max_l_c * (t / T)
        self.model.texture_swap_loss_weight.assign(l_t)
        self.model.structure_swap_loss_weight.assign(l_s)
        self.model.warp_loss_weight.assign(l_w)

    def on_batch_end(self, batch, logs):
        self.step += 1
