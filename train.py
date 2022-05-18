#!/usr/bin/env python3

import coordgan
import tensorflow as tf

def mapping_net(channels=512, layers=8, normalize=True, lrmul=0.01):
    def normalize(x, eps=1e-8):
        norm = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(norm + eps)
    
    net = tf.keras.Sequential()
    if normalize:
        net.add(tf.keras.layers.Lambda(normalize))
    for _ in range(layers):
        net.add(coordgan.layers.Dense(
            channels, lrmul=lrmul, activation='leaky_relu'))
    return net

texture_net = mapping_net()
structure_net = mapping_net()

warp_net = coordgan.models.CoordinateWarpingNetwork(
    512,
    activation = 'relu',
    image_size = (64,64),
    chamfer_loss_weight = 100.
)

mod_generator = coordgan.models.ModulatedGenerator(
    256, 4,
    image_size = (64,64),
    positional_encoding = coordgan.layers.PositionalEncoding(512),
    kernel_size = 1
)

discriminator = coordgan.models.ImageDiscriminator(
    tf.keras.Sequential([
        coordgan.layers.Conv2D(256, 1, 1, 'SAME', activation='leaky_relu'),
        coordgan.layers.ResidualBlock(256, downsample=True),
        coordgan.layers.ResidualBlock(256, downsample=True),
        coordgan.layers.ResidualBlock(256, 512, downsample=True),
        coordgan.layers.ResidualBlock(512, 1024, downsample=True),
        coordgan.layers.MinibatchStddev(group_size=4),
        coordgan.layers.Conv2D(1024, 3, 1, 'SAME', activation='leaky_relu'),
        coordgan.layers.Conv2D(1024, 4, 4, 'SAME', activation='leaky_relu'),
        coordgan.layers.Conv2D(1, 1, 1, 'SAME')
    ])
)

patch_discriminator = coordgan.models.PatchDiscriminator(
    tf.keras.Sequential([
        coordgan.layers.Conv2D(256, 1, 1, 'SAME', activation='leaky_relu'),
        coordgan.layers.ResidualBlock(256, downsample=True),
        coordgan.layers.ResidualBlock(256, downsample=True),
        coordgan.layers.ResidualBlock(256, 512, downsample=True),
        coordgan.layers.Conv2D(512, 2, 2, 'SAME', activation='leaky_relu'),
    ]),
    tf.keras.Sequential([
        coordgan.layers.Dense(512, activation='leaky_relu'),
        coordgan.layers.Dense(512, activation='leaky_relu'),
        coordgan.layers.Dense(1)
    ]),
    min_crop_size = [8,8],
    max_crop_size = [16,16],
    patch_size = [16,16]
)

gan = coordgan.CoordGAN(
    texture_net,
    structure_net,
    warp_net,
    mod_generator,
    discriminator,
    patch_discriminator,
    r1_regularization_weight = 10.,
    patch_r1_regularization_weight = 1.,
    texture_code_dim = 512,
    structure_code_dim = 512,
    gan_loss_weight = 2.0,
    texture_swap_loss_weight = 5.,
    structure_swap_loss_weight = 1.,
    warp_loss_weight = 5.,
    warp_loss_temp = 0.015**2. * 0.5)

gan.compile(
    d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.0, beta_2=0.99),
    g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.0, beta_2=0.99))

gan.fit(
    coordgan.datasets.normalized_celeb_a(64).batch(16, drop_remainder=True),
    epochs = 50,
    callbacks = [coordgan.train.LossWeightScheduler()])
