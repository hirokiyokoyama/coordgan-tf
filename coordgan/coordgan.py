import tensorflow as tf
from . import losses

class CoordGAN(tf.keras.Model):
    def __init__(self,
            texture_net,
            structure_net,
            warp_net,
            mod_generator,
            discriminator,
            patch_discriminator,
            lpips = losses.lpips,
            r1_regularization_weight = 10.,
            patch_r1_regularization_weight = 1.,
            texture_code_dim = 512,
            structure_code_dim = 512,
            gan_loss_weight = 2.,
            texture_swap_loss_weight = 5.,
            structure_swap_loss_weight = 1.,
            warp_loss_weight = 5.,
            warp_loss_img_size = None,
            warp_loss_temp = 0.015**2. * 0.5):
        super().__init__()
        self.texture_net = texture_net
        self.structure_net = structure_net
        self.warp_net = warp_net
        self.mod_generator = mod_generator
        self.discriminator = discriminator
        self.patch_discriminator = patch_discriminator
        self.warp_loss_img_size = warp_loss_img_size
        self.warp_loss_temp = warp_loss_temp
        self.lpips = lpips

        self.texture_code_dim = texture_code_dim
        self.structure_code_dim = structure_code_dim

        self.r1_regularization_weight = r1_regularization_weight
        self.patch_r1_regularization_weight = patch_r1_regularization_weight

        self.gan_loss_weight = self.add_weight(
            'gan_loss_weight',
            shape = [],
            initializer = tf.keras.initializers.Constant(gan_loss_weight),
            trainable = False)
        self.texture_swap_loss_weight = self.add_weight(
            'texture_swap_loss_weight',
            shape = [],
            initializer = tf.keras.initializers.Constant(texture_swap_loss_weight),
            trainable = False)
        self.structure_swap_loss_weight = self.add_weight(
            'structure_swap_loss_weight',
            shape = [],
            initializer = tf.keras.initializers.Constant(structure_swap_loss_weight),
            trainable = False)
        self.warp_loss_weight = self.add_weight(
            'warp_loss_weight',
            shape = [],
            initializer = tf.keras.initializers.Constant(warp_loss_weight),
            trainable = False)            

    def generate_texture_code(self, batch_size):
        z = tf.random.normal([batch_size, self.texture_code_dim])
        return z

    def generate_structure_code(self, batch_size):
        z = tf.random.normal([batch_size, self.structure_code_dim])
        return z
    
    def generate_correspondence_map(self, v_struct, training=None):
        w_struct = self.structure_net(v_struct, training=training)
        corr_map = self.warp_net(w_struct, training=training)
        return corr_map

    def generate_images(self, v_tex, v_struct, training=None):
        corr_map = self.generate_correspondence_map(v_struct, training=training)
        w_tex = self.texture_net(v_tex, training=training)
        return self.mod_generator(w_tex, coords=corr_map, training=training)

    def compile(self, d_optimizer, g_optimizer, adv_loss_fn=None):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        if adv_loss_fn is None:
            adv_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.adv_loss_fn = adv_loss_fn

    def warp_with_correspondence(self,
                src_img, dest_img, src_corr, dest_corr, temperature=None):
        N,H,W,C = tf.unstack(tf.shape(src_img))
        src_corr = tf.reshape(src_corr, [N,H*W,2])
        dest_corr = tf.reshape(dest_corr, [N,H*W,2])
        #weights = tf.matmul(dest_corr, src_corr, transpose_b=True)
        weights = tf.reduce_sum(tf.square(dest_corr[:,:,tf.newaxis,:] - src_corr[:,tf.newaxis,:,:]), axis=-1)
        weights = -weights / 2.
        if temperature is None:
            temperature = self.warp_loss_temp
        weights /= temperature
        weights = tf.nn.softmax(weights, axis=2)

        src_img = tf.reshape(src_img, [N,H*W,-1])
        src_img_warp = tf.matmul(weights, src_img)
        src_img_warp = tf.reshape(src_img_warp, [N,H,W,-1])
        return src_img_warp
    
    def d_step(self, real):
        N = tf.shape(real)[0]
        n = tf.maximum(N // 2, 1)
    
        texture_code = tf.tile(self.generate_texture_code(n), [2,1])
        structure_code = self.generate_structure_code(2*n)
        fake = self.generate_images(texture_code, structure_code, training=True)

        with tf.GradientTape() as tape:
            d_real, grad_gp = self.discriminator(real, return_grad=True, training=True)
            d_fake = self.discriminator(fake, training=True)
            gp_loss = tf.reduce_mean(tf.reduce_sum(tf.square(grad_gp), axis=[1,2,3]))
            gp_loss /= 2.
            d_loss = self.adv_loss_fn(tf.ones_like(d_real), d_real)
            d_loss += self.adv_loss_fn(tf.zeros_like(d_fake), d_fake)

            # patch discriminator (for structure swapping loss)
            pd_real, grad_gp = self.patch_discriminator(fake[0::2], fake[0::2], return_grad=True, training=True)
            pd_fake = self.patch_discriminator(fake[0::2], fake[1::2], training=True)
            pd_gp_loss = tf.reduce_mean(tf.reduce_sum(tf.square(grad_gp), axis=[1,2,3]))
            pd_gp_loss /= 2.
            struct_loss = self.adv_loss_fn(tf.ones_like(pd_real), pd_real)
            struct_loss += self.adv_loss_fn(tf.zeros_like(pd_fake), pd_fake)

            loss = d_loss + gp_loss * self.r1_regularization_weight
            loss += (struct_loss + pd_gp_loss * self.patch_r1_regularization_weight) * self.structure_swap_loss_weight
            if self.discriminator.losses:
                loss += tf.add_n(self.discriminator.losses)
            if self.patch_discriminator.losses:
                loss += tf.add_n(self.patch_discriminator.losses)
        vars = self.discriminator.trainable_variables
        vars += self.patch_discriminator.trainable_variables
        grads = tape.gradient(loss, vars)
        self.d_optimizer.apply_gradients(zip(grads, vars))

        return {'d_loss': d_loss, 'gp_loss': gp_loss, 'patch_d_loss': struct_loss}
    
    def g_step(self, real):
        N = tf.shape(real)[0]
        n = tf.maximum(N // 2, 1)

        texture_code = self.generate_texture_code(n)
        texture_code = tf.reshape(tf.tile(texture_code, [1,2]), [2*n,-1])
        structure_code = tf.tile(self.generate_structure_code(n), [2,1])
        with tf.GradientTape() as tape:
            corr_map = self.generate_correspondence_map(structure_code, training=True)
            w_tex = self.texture_net(texture_code, training=True)
            fake = self.mod_generator(w_tex, coords=corr_map, training=True)

            d_outputs = self.discriminator(fake, training=True)
            g_loss = self.adv_loss_fn(tf.ones_like(d_outputs), d_outputs)
            # texture swapping loss
            tex_loss = tf.reduce_mean(self.lpips(fake[:n], fake[n:]))
            # structure swapping loss
            pd_outputs = self.patch_discriminator(fake[0::2], fake[1::2], training=True)
            struct_loss = self.adv_loss_fn(tf.ones_like(pd_outputs), pd_outputs)
            # warping loss
            fake1 = fake[::2]
            fake2 = fake[::-2]
            corr1 = corr_map[::2]
            corr2 = corr_map[::-2]
            if self.warp_loss_img_size is not None:
                fake1 = tf.image.resize(fake1, self.warp_loss_img_size)
                fake2 = tf.image.resize(fake2, self.warp_loss_img_size)
                corr1 = tf.image.resize(corr1, self.warp_loss_img_size)
                corr2 = tf.image.resize(corr2, self.warp_loss_img_size)
            fake1_warp = self.warp_with_correspondence(
                fake1, fake2, corr1, corr2, self.warp_loss_temp)
            warp_loss = tf.reduce_mean(self.lpips(fake2, fake1_warp))

            loss = g_loss * self.gan_loss_weight
            loss += tex_loss * self.texture_swap_loss_weight + struct_loss * self.structure_swap_loss_weight
            loss += warp_loss * self.warp_loss_weight
            if self.mod_generator.losses:
                loss += tf.add_n(self.mod_generator.losses)
            if self.texture_net.losses:
                loss += tf.add_n(self.texture_net.losses)
            if self.structure_net.losses:
                loss += tf.add_n(self.structure_net.losses)
            if self.warp_net.losses:
                loss += tf.add_n(self.warp_net.losses)
        vars = self.mod_generator.trainable_variables
        vars += self.texture_net.trainable_variables
        vars += self.structure_net.trainable_variables
        vars += self.warp_net.trainable_variables
        grads = tape.gradient(loss, vars)
        self.g_optimizer.apply_gradients(zip(grads, vars))        

        return {
            'g_loss': g_loss,
            'tex_swap_loss': tex_loss,
            'struct_swap_loss': struct_loss,
            'warp_loss': warp_loss
        }

    def train_step(self, real):
        d_losses = self.d_step(real)
        g_losses = self.g_step(real)

        d_losses.update(g_losses)
        return d_losses

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.patch_discriminator.save_weights(filepath+"/patch_discriminator/weights", overwrite, save_format, options)
        self.discriminator.save_weights(filepath+"/discriminator/weights", overwrite, save_format, options)
        self.mod_generator.save_weights(filepath+"/mod_generator/weights", overwrite, save_format, options)
        self.texture_net.save_weights(filepath+"/texture_net/weights", overwrite, save_format, options)
        self.structure_net.save_weights(filepath+"/structure_net/weights", overwrite, save_format, options)
        self.warp_net.save_weights(filepath+"/warp_net/weights", overwrite, save_format, options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.patch_discriminator.load_weights(filepath+"/patch_discriminator/weights", by_name, skip_mismatch, options)
        self.discriminator.load_weights(filepath+"/discriminator/weights", by_name, skip_mismatch, options)
        self.mod_generator.load_weights(filepath+"/mod_generator/weights", by_name, skip_mismatch, options)
        self.texture_net.load_weights(filepath+"/texture_net/weights", by_name, skip_mismatch, options)
        self.structure_net.load_weights(filepath+"/structure_net/weights", by_name, skip_mismatch, options)
        self.warp_net.load_weights(filepath+"/warp_net/weights", by_name, skip_mismatch, options)
