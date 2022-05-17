import tensorflow as tf
from . import lpips as _lpips

def chamfer(coord1, coord2, n1=None, epsilon=1e-8):
    N,H,W,C = tf.unstack(tf.shape(coord1))
    C1 = tf.reshape(coord1, [N,-1,1,C])
    C2 = tf.reshape(coord2, [N,1,-1,C])
    if n1 is not None and n1 < H*W:
        inds = tf.random.shuffle(tf.range(H*W))[:n1]
        C1 = tf.gather(C1, inds, axis=1)
    diff = tf.reduce_sum(tf.square(C1 - C2), axis=-1)
    diff = tf.math.sqrt(diff + epsilon)
    loss = tf.reduce_mean(tf.reduce_min(diff, axis=1))
    loss += tf.reduce_mean(tf.reduce_min(diff, axis=2))
    return loss

_lpips_vgg16 = None

def lpips(x, y, size=[128,128]):
    if _lpips_vgg16 is None:
        _lpips_vgg16 = _lpips.LPIPS_VGG16()
        _lpips_vgg16.load_pretrained_weights()
        _lpips_vgg16.trainable = False
        
    x = tf.image.resize(x, size)
    y = tf.image.resize(y, size)
    return _lpips_vgg16(x, y)
