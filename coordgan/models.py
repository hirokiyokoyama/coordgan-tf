import tensorflow as tf
from . import layers
from . import losses

class ModulatedGenerator(tf.keras.Model):
    def __init__(self, channels, num_blocks,
                 image_channels = 3,
                 image_size = (128,128),
                 positional_encoding = None,
                 coord_encoding_dim = None,
                 kernel_size = 3,
                 demodulate = True,
                 fused_mod_conv = False,
                 activation = 'leaky_relu',
                 output_activation = None):
        super().__init__()
        self.modulated_convs = [
            layers.ModulatedConv2D(
                channels, kernel_size, 1, 'SAME',
                fused = fused_mod_conv,
                data_format = 'channels_first',
                activation = activation,
                demodulate = demodulate) for _ in range(num_blocks * 2 + 1)
        ]

        self.output_convs = [
            layers.ModulatedConv2D(
                image_channels, 1, 1, 'SAME',
                fused = fused_mod_conv,
                data_format = 'channels_first',
                activation = output_activation,
                demodulate = False) for _ in range(num_blocks)
        ]
        self.image_size = image_size

        if coord_encoding_dim:
            self.coord_encoding = self.add_weight(
                'coord_encoding', shape=[*image_size,coord_encoding_dim],
                initializer = 'truncated_normal')
        else:
            self.coord_encoding = None

        if positional_encoding is not None:
            self.positional_encoding = positional_encoding
        else:
            self.positional_encoding = layers.PositionalEncoding(channels)
    
    def call(self, style, coords=None):
        if coords is None:
            I = tf.range(self.image_size[0], dtype=self.dtype) + 0.5
            I = (I / tf.cast(self.image_size[0], dtype=self.dtype)) * 2. - 1.
            J = tf.range(self.image_size[1], dtype=self.dtype) + 0.5
            J = (J / tf.cast(self.image_size[1], dtype=self.dtype)) * 2. - 1.
            coords = tf.stack(tf.meshgrid(I, J, indexing='ij'), axis=-1)
            n = tf.shape(style)[0]
            coords = tf.tile(coords[tf.newaxis], [n,1,1,1])

        x = self.positional_encoding(coords)
        
        if self.coord_encoding is not None:
            _x = self.coord_encoding[tf.newaxis]
            _x = tf.tile(_x, [tf.shape(style)[0],1,1,1])
            x = tf.concat([x, _x], axis=-1)

        x = tf.transpose(x, [0,3,1,2])
        coords = tf.transpose(coords, [0,3,1,2])

        x = self.modulated_convs[0](x, style)
        y = 0.
        for b in range(len(self.modulated_convs) // 2):
            conv1 = self.modulated_convs[b*2+1]
            conv2 = self.modulated_convs[b*2+2]

            _x = x
            _x = tf.concat([_x, coords], axis=1)
            _x = conv1(_x, style)
            _x = tf.concat([_x, coords], axis=1)
            _x = conv2(_x, style)
            x = _x

            _y = tf.concat([_x, coords], axis=1)
            _y = self.output_convs[b](_y, style)
            y += _y

        y = tf.transpose(y, [0,2,3,1])
        return y

class CoordinateWarpingNetwork(tf.keras.Model):
    def __init__(self, hidden_channels,
                 image_size = (128,128),
                 activation = 'relu',
                 chamfer_loss_weight = 100.,
                 chamfer_sample_num = None):
        super().__init__()
        self.conv1 = layers.Conv2D(
            hidden_channels, 1, 1, 'SAME', activation=activation)
        self.conv2 = layers.Conv2D(2, 1, 1, 'SAME', activation='tanh')

        self.image_size = image_size
        self.chamfer_loss_weight = self.add_weight(
            'chamfer_loss_weight',
            shape = [],
            initializer = tf.keras.initializers.Constant(chamfer_loss_weight),
            trainable = False)
        self.chamfer_sample_num = chamfer_sample_num

    def build(self, shape):
        self.coord_conv = layers.Conv2D(
            shape[-1], 1, 1, 'SAME', use_bias=False)
        self.built = True

    def call(self, struct, coords=None):
        if coords is None:
            H, W = self.image_size
            N = tf.shape(struct)[0]
            I = tf.range(H, dtype=self.dtype) + 0.5
            I = (I / tf.cast(H, self.dtype)) * 2. - 1.
            J = tf.range(W, dtype=self.dtype) + 0.5
            J = (J / tf.cast(W, self.dtype)) * 2. - 1.
            coords = tf.stack(tf.meshgrid(I, J, indexing='ij'), axis=-1)
            coords = tf.tile(coords[tf.newaxis], [N,1,1,1])
        else:
            shape = tf.shape(coords)
            N = shape[0]
            H = shape[1]
            W = shape[2]

        w = tf.tile(struct[:,tf.newaxis,tf.newaxis,:], [1,H,W,1])
        _coords = self.coord_conv(coords)
        x = self.conv1(tf.concat([w, _coords], axis=-1))
        x = self.conv2(x)
        
        if self.chamfer_loss_weight > 0.:
            ch_loss = losses.chamfer(coords, x, n1=self.chamfer_sample_num)
        else:
            ch_loss = 0.
        self.add_loss(ch_loss * self.chamfer_loss_weight)
        return x
    
class ImageDiscriminator(tf.keras.Model):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def call(self, images, return_grad=False):
        def _call(x):
            x = self.classifier(x)
            if x.shape.rank == 4:
                x = tf.squeeze(x, [1,2])
            return x
                
        if not return_grad:
            return _call(images)

        with tf.GradientTape() as tape:
            tape.watch(images)
            d = _call(images)
        grad = tape.gradient(d, images)
        return d, grad
        
class PatchDiscriminator(tf.keras.Model):
    def __init__(self, patch_encoder, classifier,
                 num_patches = 8,
                 min_crop_size = [16, 16],
                 max_crop_size = [32, 32],
                 patch_size = [32, 32]):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.classifier = classifier
        self.num_patches = num_patches
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.patch_size = patch_size
    
    def extract_patches(self, images):
        patches = []
        H, W = tf.unstack(tf.shape(images)[1:3])
        for _ in range(self.num_patches):
            h = tf.random.uniform([], self.min_crop_size[0], self.max_crop_size[0]+1, dtype=tf.int32)
            w = tf.random.uniform([], self.min_crop_size[1], self.max_crop_size[1]+1, dtype=tf.int32)
            top = tf.random.uniform([], 0, H-h+1, dtype=tf.int32)
            left = tf.random.uniform([], 0, W-w+1, dtype=tf.int32)
            patch = images[:,top:top+h,left:left+w,:]
            patch = tf.image.resize(patch, self.patch_size)
            patches.append(patch)
        return tf.stack(patches, axis=1)
    
    def call(self, reference, target, return_grad=False):
        # extract patches from reference 
        reference = self.extract_patches(reference)
        N,P,H,W,C = tf.unstack(tf.shape(reference))
        reference = tf.reshape(reference, [N*P,H,W,C])

        # extract patches from target
        target = self.extract_patches(target)
        N,P,H,W,C = tf.unstack(tf.shape(target))
        target = tf.reshape(target, [N*P,H,W,C])

        def _call(ref, x):
            ref = self.patch_encoder(ref)
            ref = tf.reshape(ref, [N,P,-1])
            ref = tf.reduce_mean(ref, axis=1)

            x = self.patch_encoder(x)
            x = tf.reshape(x, [N*P,-1])
            ref = tf.tile(ref[:,tf.newaxis,:], [1,P,1])
            ref = tf.reshape(ref, [N*P,-1])
            d = self.classifier(tf.concat([ref, x], axis=1))
            d = tf.reshape(d, [N,P,1])
            return d

        if not return_grad:
            return _call(reference, target)

        with tf.GradientTape() as tape:
            tape.watch(target)
            #tape.watch(reference)
            d = _call(reference, target)
        grad = tape.gradient(d, target)
        #grad = tape.gradient(d, reference)
        return d, grad
