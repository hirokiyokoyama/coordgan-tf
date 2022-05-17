import tensorflow as tf

VGG_TORCH_URL = 'https://download.pytorch.org/models/vgg16-397923af.pth'
LPIPS_VGG_URL = 'https://github.com/alexlee-gk/PerceptualSimilarity/raw/179dc5c8c2fbf4056e80f480adff4bc17d623faf/weights/v0.1/vgg.pth'

class LPIPS_VGG16(tf.keras.Model):
    def __init__(self, epsilon=1e-10):
        super().__init__()

        self.conv1_1 = tf.keras.layers.Conv2D(
            64, 3, activation='relu', padding='same', name='block1_conv1')
        self.conv1_2 = tf.keras.layers.Conv2D(
            64, 3, activation='relu', padding='same', name='block1_conv2')
        self.pool1 = tf.keras.layers.MaxPooling2D(2, strides=2, name='block1_pool')

        self.conv2_1 = tf.keras.layers.Conv2D(
            128, 3, activation='relu', padding='same', name='block2_conv1')
        self.conv2_2 = tf.keras.layers.Conv2D(
            128, 3, activation='relu', padding='same', name='block2_conv2')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=2, name='block2_pool')

        self.conv3_1 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same', name='block3_conv1')
        self.conv3_2 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same', name='block3_conv2')
        self.conv3_3 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same', name='block3_conv3')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, strides=2, name='block3_pool')

        self.conv4_1 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same', name='block4_conv1')
        self.conv4_2 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same', name='block4_conv2')
        self.conv4_3 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same', name='block4_conv3')
        self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, name='block4_pool')

        self.conv5_1 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same', name='block5_conv1')
        self.conv5_2 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same', name='block5_conv2')
        self.conv5_3 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same', name='block5_conv3')

        self.lin1 = tf.keras.layers.Conv2D(
            1, 1, padding='same', use_bias=False, name='lin1')
        self.lin2 = tf.keras.layers.Conv2D(
            1, 1, padding='same', use_bias=False, name='lin2')
        self.lin3 = tf.keras.layers.Conv2D(
            1, 1, padding='same', use_bias=False, name='lin3')
        self.lin4 = tf.keras.layers.Conv2D(
            1, 1, padding='same', use_bias=False, name='lin4')
        self.lin5 = tf.keras.layers.Conv2D(
            1, 1, padding='same', use_bias=False, name='lin5')
        
        self.epsilon = epsilon

    def preprocess(self, x):
        mean = tf.constant([-.030, -.088, -.188], self.dtype)
        std = tf.constant([.458, .448, .450], self.dtype)
        return (x - mean) / std

    def vgg_features(self, x):
        features = []
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        features.append(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        features.append(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        features.append(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        features.append(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        features.append(x)
        return features

    def call(self, x1, x2):
        x = tf.concat([x1, x2], axis=0)
        x = self.preprocess(x)
        features = self.vgg_features(x)
        lins = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5]
        y = 0.
        for feat, lin in zip(features, lins):
            norm = tf.norm(feat, axis=-1, keepdims=True)
            feat /= (norm + self.epsilon)
            feat1, feat2 = tf.split(feat, 2, axis=0)
            diff = tf.square(feat1 - feat2)
            diff = lin(diff)
            y += tf.reduce_mean(diff, axis=[1,2])
        return y

    def load_vgg_weights(self):
        import torch
        vgg_file = VGG_TORCH_URL.split('/')[-1]
        vgg_path = tf.keras.utils.get_file(
            vgg_file, VGG_TORCH_URL, cache_subdir='models')
        with open(vgg_path, 'rb') as f:
            x = torch.load(f, map_location=torch.device('cpu'))

        def load_conv(conv, key):
            W = x[f'{key}.weight'].permute(2,3,1,0)
            b = x[f'{key}.bias']
            conv.kernel.assign(W)
            conv.bias.assign(b)
        def load_dense(dense, key):
            W = x[f'{key}.weight'].permute(1,0)
            b = x[f'{key}.bias']
            dense.kernel.assign(W)
            dense.bias.assign(b)
        load_conv(self.conv1_1, 'features.0')
        load_conv(self.conv1_2, 'features.2')
        load_conv(self.conv2_1, 'features.5')
        load_conv(self.conv2_2, 'features.7')
        load_conv(self.conv3_1, 'features.10')
        load_conv(self.conv3_2, 'features.12')
        load_conv(self.conv3_3, 'features.14')
        load_conv(self.conv4_1, 'features.17')
        load_conv(self.conv4_2, 'features.19')
        load_conv(self.conv4_3, 'features.21')
        load_conv(self.conv5_1, 'features.24')
        load_conv(self.conv5_2, 'features.26')
        load_conv(self.conv5_3, 'features.28')

    def load_lin_weights(self):
        import torch
        lin_file = LPIPS_VGG_URL.split('/')[-1]
        lin_path = tf.keras.utils.get_file(
            lin_file, LPIPS_VGG_URL, cache_subdir='models')
        with open(lin_path, 'rb') as f:
            x = torch.load(f, map_location=torch.device('cpu'))
        def load_conv(conv, key):
            W = x[f'{key}.weight'].permute(2,3,1,0)
            conv.kernel.assign(W)
        load_conv(self.lin1, 'lin0.model.1')
        load_conv(self.lin2, 'lin1.model.1')
        load_conv(self.lin3, 'lin2.model.1')
        load_conv(self.lin4, 'lin3.model.1')
        load_conv(self.lin5, 'lin4.model.1')

    def load_pretrained_weights(self):
        self(tf.keras.Input([None,None,3]), tf.keras.Input([None,None,3]))
        self.load_vgg_weights()
        self.load_lin_weights()
