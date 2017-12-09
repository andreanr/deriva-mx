import csv
import sys
import requests
import skimage.io
import os
import pdb
import glob
import pickle
import time

from IPython.display import display, Image, HTML
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.misc import imsave

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def get_kept_filters(layer_dict,
                    layer_name,
                    input_img,
                    image_width,
                    image_height,
                    m):
    kept_filters = []
    for filter_index in range(m):

        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    n = len(kept_filters[0][0][0][0])
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                        (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave('{layer}_{n}x{n}.png'.format(layer=layer_name, n=n), stitched_filters)

if __name__ == '__main__':
    img_width = 128
    img_height = 128

    # build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=False)
    print(model.summary())

    # using ImageNet images (tensor)
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names)
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    layer_name = 'block5_conv3'
    m = 200
    get_kept_filters(layer_dict,
                     layer_name,
                     input_img,
                     img_width,
                     img_height,
                     m)
