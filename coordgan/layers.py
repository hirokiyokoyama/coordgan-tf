import tensorflow as tf
import math

from . import losses

class Conv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise ValueError('kernel_initializer cannot be specified.')
        kwargs['kernel_initializer'] = tf.keras.initializers.RandomNormal(0., 1.)
        super().__init__(*args, **kwargs)

    def build(self, shape):
        super().build(shape)
        fan_in = tf.cast(tf.reduce_prod(self.kernel.shape[:-1]), self.dtype)
        self.scale = tf.math.sqrt(2. / fan_in)

    def convolution_op(self, inputs, kernel):
        return super().convolution_op(inputs, kernel) * self.scale
    
class Dense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise ValueError('kernel_initializer cannot be specified.')
        lrmul = kwargs.pop('lrmul', 1.)
        kwargs['kernel_initializer'] = tf.keras.initializers.RandomNormal(0., 1./lrmul)
        super().__init__(*args, **kwargs)
        self.lrmul = lrmul

    def build(self, shape):
        super().build(shape)
        fan_in = tf.cast(tf.reduce_prod(self.kernel.shape[:-1]), self.dtype)
        self.scale = tf.math.sqrt(2. / fan_in)
        self._use_bias = self.use_bias
        self.use_bias = False
        self._activation = self.activation
        self.activation = None

    def call(self, x):
        x = super().call(x * self.scale)
        if self._use_bias:
            x += self.bias
        x *= self.lrmul
        if self._activation is not None:
            x = self._activation(x)
        return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, channels, frequency_scale=1.):
        super().__init__()
        
        scale = frequency_scale
        # fan_out -> fan_in
        scale *= math.sqrt(channels / 2.)
        self.scale = scale

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            mode='fan_out', distribution='uniform')
        self.conv = tf.keras.layers.Conv2D(
            channels, 1, 1, 'SAME',
            kernel_initializer = kernel_initializer,
            activation = tf.math.sin)

    def call(self, x):
        return self.conv(x * self.scale)
        
class ModulatedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise ValueError('kernel_initializer cannot be specified.')
        kwargs['kernel_initializer'] = tf.keras.initializers.RandomNormal(0., 1.)

        self.epsilon = kwargs.pop('epsilon', 1e-8)
        self.demodulate = kwargs.pop('demodulate', True)
        self.fused = kwargs.pop('fused', False)
        super().__init__(*args, **kwargs)

    def build(self, shape):
        super().build(shape)
        c = shape[-1] if self.data_format == 'channels_last' else shape[1]
        self.mod_dense = Dense(c, bias_initializer='ones')
        fan_in = tf.cast(tf.reduce_prod(self.kernel.shape[:-1]), self.dtype)
        self.scale = tf.math.sqrt(2. / fan_in)

    def convolution_op(self, inputs, kernel):
        if self.fused:
            kernel = tf.transpose(self._scaled_kernel, [1,2,3,0,4])
            kernel = tf.reshape(kernel, [*self.kernel.shape[:3], -1])

            if self.data_format == 'channels_last':
                N,H,W,C = tf.unstack(tf.shape(inputs))
                x = tf.reshape(tf.transpose(inputs, [1,2,0,3]), [1,H,W,-1])
                x = super().convolution_op(x, kernel)
                x = tf.transpose(tf.reshape(x, [H,W,N,-1]), [2,0,1,3])
            else:
                N,C,H,W = tf.unstack(tf.shape(inputs))
                x = tf.reshape(inputs, [1,-1,H,W])
                x = super().convolution_op(x, kernel)
                x = tf.reshape(x, [N,-1,H,W])
        else:
            if self.data_format == 'channels_last':
                x = inputs * self._pre_scale[:,tf.newaxis,tf.newaxis,:]
                x = super().convolution_op(x, kernel)
                x = x * self._post_scale[:,tf.newaxis,tf.newaxis,:]
            else:
                x = inputs * self._pre_scale[:,:,tf.newaxis,tf.newaxis]
                x = super().convolution_op(x, kernel)
                x = x * self._post_scale[:,:,tf.newaxis,tf.newaxis]
        return x

    def call(self, x, y):
        s = self.mod_dense(y)
        self._pre_scale = s

        if self.demodulate:
            w = self.kernel[tf.newaxis] * s[:,tf.newaxis,tf.newaxis,:,tf.newaxis]
            _w = tf.reduce_sum(tf.square(w), axis=[1,2,3])
            d = tf.math.rsqrt(_w + self.epsilon)
            self._post_scale = d
            if self.fused:
                self._scaled_kernel = w * d[:,tf.newaxis,tf.newaxis,tf.newaxis,:]
        else:
            self._post_scale = self.scale[tf.newaxis,tf.newaxis]
            if self.fused:
                w = self.kernel[tf.newaxis] * s[:,tf.newaxis,tf.newaxis,:,tf.newaxis]
                self._scaled_kernel = w * self.scale

        x = super().call(x)
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, output_channels=None, downsample=False, activation='leaky_relu'):
        super().__init__()
        if output_channels is None:
            output_channels = channels

        self.conv1 = Conv2D(channels, 3, 1, 'SAME', activation=activation)
        self.conv2 = Conv2D(output_channels, 3, 2 if downsample else 1, 'SAME', activation=activation)
        self.skip_conv = Conv2D(output_channels, 1, 2 if downsample else 1, 'SAME', use_bias=False)
        self.downsample = downsample

        k = tf.constant([1., 3., 3., 1.], dtype=self.dtype)
        k = k[tf.newaxis,:] * k[:,tf.newaxis]
        k /= tf.reduce_sum(k)
        self.resample_kernel = k[:,:,tf.newaxis,tf.newaxis]
    
    def resample(self, x):
        if self.downsample:
            C = x.shape[-1]
            k = tf.tile(self.resample_kernel, [1,1,C,1])
            x = tf.nn.depthwise_conv2d(x, k, [1,1,1,1], 'SAME')
        return x

    def call(self, x):
        y = x
        y = self.conv1(y)
        y = self.resample(y)
        y = self.conv2(y)
        x = self.skip_conv(self.resample(x))
        return (y + x) / math.sqrt(2.)

class MinibatchStddev(tf.keras.layers.Layer):
    def __init__(self, group_size=4, num_new_features=1, epsilon=1e-8):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features
        self.epsilon = epsilon

    def call(self, x):
        N, h, w, c = tf.unstack(tf.shape(x))
        group_size = tf.minimum(self.group_size, N)
        n = self.num_new_features

        # [GMHWnc]
        y = tf.reshape(x, [group_size, -1, h, w, n, c//n])
        # [GMHWnc] Subtract mean over group.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        # [MHWnc]  Calc variance over group.
        y = tf.reduce_mean(tf.square(y), axis=0)
        # [MHWnc]  Calc stddev over group.
        y = tf.sqrt(y + self.epsilon)
        # [M11n1]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[1,2,4], keepdims=True)
        # [M11n]
        y = tf.squeeze(y, -1)
        # [NHWn]  Replicate over group and pixels.
        y = tf.tile(y, [group_size, h, w, 1])
        # [NHWC]  Append as new fmap.
        return tf.concat([x, y], axis=-1)
