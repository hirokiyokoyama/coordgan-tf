import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

def rot90(v):
    return tf.stack([v[...,1], -v[...,0]], axis=-1)

def gaussian_blur(image, sigma):
    sigma = tf.convert_to_tensor(sigma)
    if tf.rank(sigma) == 0:
        sigma = tf.stack([sigma, sigma])
    ksize = tf.cast(tf.math.ceil(2. * sigma), tf.int32)
    if tf.reduce_max(ksize) <= 1:
        return image

    Y, X = tf.meshgrid(
        tf.linspace(-ksize[0], ksize[0], ksize[0] * 2 + 1,),
        tf.linspace(-ksize[1], ksize[1], ksize[1] * 2 + 1), indexing='ij')
    Y = tf.cast(Y, sigma.dtype)
    X = tf.cast(X, sigma.dtype)
    kernel = tf.math.exp(-0.5 * ((Y / sigma[0])**2 + (X / sigma[1])**2))
    kernel /= tf.reduce_sum(kernel)
    kernel = kernel[:,:,tf.newaxis,tf.newaxis]
    image = tf.transpose(image[tf.newaxis], [3,1,2,0])
    image = tf.nn.conv2d(image, kernel, [1,1,1,1], 'SAME')
    image = tf.transpose(image, [3,1,2,0])[0]
    return image

def solve_homography(p1, p2):
    p1 = tf.convert_to_tensor(p1)
    p2 = tf.convert_to_tensor(p2)
    if p1.shape[0] != 4 or p2.shape[0] != 4:
        raise ValueError('Four pairs of points must be specified.')

    xy1 = tf.concat([p1, tf.ones([4, 1])], axis=1)
    XY1 = tf.concat([p2, tf.ones([4, 1])], axis=1)
    A1 = tf.concat([tf.zeros([4, 3]), xy1, -p2[:,1:2] * p1], axis=1)
    A2 = tf.concat([xy1, tf.zeros([4, 3]), -p2[:,0:1] * p1], axis=1)
    A = tf.concat([A1, A2], axis=0)
    b = tf.concat([p2[:,1:2], p2[:,0:1]], axis=0)
    return tf.linalg.solve(A, b)[:,0]

def normalize_celeb_a(x, target_size=128):
    image = tf.cast(x['image'], tf.float32) / 255.
    image_size = tf.shape(image)[:2]

    lm = x['landmarks']
    leye = tf.cast(tf.stack([lm['lefteye_y'], lm['lefteye_x']]), tf.float32)
    reye = tf.cast(tf.stack([lm['righteye_y'], lm['righteye_x']]), tf.float32)
    lmouth = tf.cast(tf.stack([lm['leftmouth_y'], lm['leftmouth_x']]), tf.float32)
    rmouth = tf.cast(tf.stack([lm['rightmouth_y'], lm['rightmouth_x']]), tf.float32)
    nose = tf.cast(tf.stack([lm['nose_y'], lm['nose_x']]), tf.float32)
    lm = tf.stack([leye, reye, lmouth, rmouth, nose], axis=0)

    eye_avg = (leye + reye) * 0.5 + 0.5
    mouth_avg = (lmouth + rmouth) * 0.5 + 0.5
    eye_to_eye = reye - leye
    eye_to_mouth = mouth_avg - eye_avg
    qx = eye_to_eye - rot90(eye_to_mouth)
    qx /= tf.linalg.norm(qx)
    qx *= tf.maximum(tf.linalg.norm(eye_to_eye) * 2.0, tf.linalg.norm(eye_to_mouth) * 1.8)
    qy = rot90(qx)
    c = eye_avg + eye_to_mouth * 0.1
    quad = tf.stack([c - qx - qy, c - qx + qy, c + qx + qy, c + qx - qy])
    zoom = target_size / (tf.linalg.norm(qx) * 2.)

    # crop if possible
    border = tf.maximum(tf.cast(tf.math.round(target_size * 0.1 / zoom), tf.int32), 3)
    crop_tl = tf.cast(tf.math.floor(tf.reduce_min(quad, axis=0)), tf.int32)
    crop_br = tf.cast(tf.math.ceil(tf.reduce_max(quad, axis=0)), tf.int32)
    crop_tl = tf.maximum(crop_tl - border, 0)
    crop_br = tf.minimum(crop_br + border, image_size)
    if tf.reduce_any(crop_br - crop_tl < image_size):
        image = image[crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1]]
        quad -= tf.cast(crop_tl, quad.dtype)
        lm -= tf.cast(crop_tl, lm.dtype)
        image_size = tf.shape(image)[:2]

    # pad if needed
    pad_tl = tf.cast(tf.math.floor(tf.reduce_min(quad, axis=0)), tf.int32)
    pad_br = tf.cast(tf.math.ceil(tf.reduce_max(quad, axis=0)), tf.int32)
    pad_tl = tf.maximum(-pad_tl + border, 0)
    pad_br = tf.maximum(pad_br - image_size + border, 0)

    if tf.maximum(tf.reduce_max(pad_tl), tf.reduce_max(pad_br)) > border - 4:
        min_pad = tf.cast(tf.math.round(target_size * 0.3 / zoom), pad_tl.dtype)
        pad_tl = tf.maximum(pad_tl, min_pad)
        pad_br = tf.maximum(pad_br, min_pad)
        image = tf.pad(image, [[pad_tl[0], pad_br[0]], [pad_tl[1], pad_br[1]], [0, 0]], mode='REFLECT')
        image_size = tf.shape(image)[:2]

        Y, X = tf.meshgrid(tf.range(image_size[0]), tf.range(image_size[1]), indexing='ij')
        mask = 1.0 - tf.minimum(
            tf.minimum(tf.cast(Y, tf.float32) / tf.cast(pad_tl[0], tf.float32), tf.cast(X, tf.float32) / tf.cast(pad_tl[1], tf.float32)),
            tf.minimum(tf.cast(image_size[0]-1-Y, tf.float32) / tf.cast(pad_br[0], tf.float32), tf.cast(image_size[1]-1-X, tf.float32) / tf.cast(pad_br[1], tf.float32)))
        mask = mask[:,:,tf.newaxis]
        blur = target_size * 0.02 / zoom
        image += (gaussian_blur(image, blur) - image) * tf.clip_by_value(mask * 3.0 + 1.0, 0.0, 1.0)
        # using mean instead of median
        image += (tf.reduce_mean(image, axis=[0,1], keepdims=True) - image) * tf.clip_by_value(mask, 0.0, 1.0)
        quad += tf.cast(pad_tl, quad.dtype)
        lm += tf.cast(pad_tl, lm.dtype)

    Hinv = solve_homography(
        [[0., 0.], [0., target_size-1.], [target_size-1., target_size-1.], [target_size-1., 0.]],
        quad[:,::-1])
    image = tfa.image.transform(
        image, Hinv, output_shape=[target_size, target_size], interpolation='bilinear')

    # transform landmarks
    Hinv = tf.reshape(tf.concat([Hinv, [1.]], 0), [3, 3])
    lm = tf.concat([tf.transpose(lm[:,::-1]), tf.ones([1,5])], axis=0)
    lm = tf.linalg.solve(Hinv, lm)
    lm = lm[:2,:] / lm[2:,:]
    lm = tf.transpose(lm)[:,::-1]
    lm = {
        'lefteye_y': lm[0,0],
        'lefteye_x': lm[0,1],
        'righteye_y': lm[1,0],
        'righteye_x': lm[1,1],
        'leftmouth_y': lm[2,0],
        'leftmouth_x': lm[2,1],
        'rightmouth_y': lm[3,0],
        'rightmouth_x': lm[3,1],
        'nose_y': lm[4,0],
        'nose_x': lm[4,1],
    }
    return {
        'image': image,
        'attributes': x['attributes'],
        'landmarks': lm
    }

def normalized_celeb_a(size, split='train', **kwargs):
    dataset = tfds.load('celeb_a', split=split, **kwargs)
    dataset = dataset.map(lambda x: normalize_celeb_a(x, size)['image'] * 2. - 1.)
    return dataset

def cropped_lfw(size=64, crop=32, split='train', **kwargs):
    dataset = tfds.load('lfw', split=split, **kwargs)
    dataset = dataset.map(lambda x: tf.cast(x['image'], tf.float32)/255.0 * 2. - 1.)
    dataset = dataset.map(lambda x: tf.image.resize(x[crop:-crop,crop:-crop], [size,size]))
    return dataset
