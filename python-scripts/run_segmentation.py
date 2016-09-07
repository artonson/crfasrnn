# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on
top of the Caffe deep learning library.

Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor:
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""
import argparse
import glob
from itertools import izip
from operator import itemgetter

caffe_root = '../caffe/'
import sys, getopt
sys.path.insert(0, caffe_root + 'python')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt


def tic():
    #http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


def run_crfasrnn(inputfile, outputfile, gpudevice):
    MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
    PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
    IMAGE_FILE = inputfile

    if gpudevice >=0:
        #Do you have GPU device?
        has_gpu = 1
        #which gpu device is available?
        gpu_device=gpudevice#assume the first gpu device is available, e.g. Titan X
    else:
        has_gpu = 0


    if has_gpu==1:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
        tic()
        net = caffe.Segmenter(MODEL_FILE, PRETRAINED,True)
        toc()
    else:
        caffe.set_mode_cpu()
        tic()
        net = caffe.Segmenter(MODEL_FILE, PRETRAINED,False)
        toc()


    input_image = 255 * caffe.io.load_image(IMAGE_FILE)


    # width = input_image.shape[0]
    # height = input_image.shape[1]
    # maxDim = max(width,height)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    pallete = [0,0,0,
            128,0,0,
            0,128,0,
            128,128,0,
            0,0,128,
            128,0,128,
            0,128,128,
            128,128,128,
            64,0,0,
            192,0,0,
            64,128,0,
            192,128,0,
            64,0,128,
            192,0,128,
            64,128,128,
            192,128,128,
            0,64,0,
            128,64,0,
            0,192,0,
            128,192,0,
            0,64,128,
            128,64,128,
            0,192,128,
            128,192,128,
            64,64,0,
            192,64,0,
            64,192,0,
            192,192,0]

    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

    # Rearrange channels to form BGR
    im = image[:,:,::-1]
    # Subtract mean
    im = im - reshaped_mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = 500 - cur_h
    pad_w = 500 - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
    # Get predictions
    segmentation = net.predict([im])
    segmentation2 = segmentation[0:cur_h, 0:cur_w]
    output_im = PILImage.fromarray(segmentation2)
    output_im.putpalette(pallete)

    plt.imshow(output_im)
    plt.savefig(outputfile)



def chunkify(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


CHUNK_SIZE = 64


MEAN_VEC = np.array([103.939, 116.779, 123.68], dtype=np.float32)
RESHAPED_MEAN_VEC = MEAN_VEC.reshape(1, 1, 3)
PALETTE = [0,0,0,
           128,0,0,
           0,128,0,
           128,128,0,
           0,0,128,
           128,0,128,
           0,128,128,
           128,128,128,
           64,0,0,
           192,0,0,
           64,128,0,
           192,128,0,
           64,0,128,
           192,0,128,
           64,128,128,
           192,128,128,
           0,64,0,
           128,64,0,
           0,192,0,
           128,192,0,
           0,64,128,
           128,64,128,
           0,192,128,
           128,192,128,
           64,64,0,
           192,64,0,
           64,192,0,
           192,192,0]
MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'


def run_crfasrnn_in_batches(input_batch, output_batch, gpu):
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    net = caffe.Segmenter(MODEL_FILE, PRETRAINED, True)

    input_images = []
    input_shapes = []
    for filename in input_batch:
        image = 255 * caffe.io.load_image(filename)
        image = PILImage.fromarray(np.uint8(image))
        image = np.array(image)

        # Rearrange channels to form BGR
        image = image[:,:,::-1]
        # Subtract mean
        image = image - RESHAPED_MEAN_VEC

        # Pad as necessary
        cur_h, cur_w, cur_c = image.shape
        input_shapes.append(image.shape)
        pad_h = 500 - cur_h
        pad_w = 500 - cur_w
        image = np.pad(image,
                       pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                       mode='constant',
                       constant_values=0)

        input_images.append(image)

    # Get predictions
    segmentations = net.predict(input_images)
    for segmentation, input_shape, output_filename in izip(segmentations, input_shapes, output_batch):
        cur_h, cur_w, cur_c = input_shape
        segmentation2 = segmentation[0:cur_h, 0:cur_w]
        output_im = PILImage.fromarray(segmentation2)
        output_im.putpalette(PALETTE)
        plt.imshow(output_im)
        plt.savefig(output_filename)


def main(options):
    files = glob.glob(os.path.join(options.input_dir, '*'))
    output_files = [os.path.join(options.output_dir, os.path.basename(filename))
                    for filename in files]
    first = itemgetter(0)
    second = itemgetter(1)
    for batch in chunkify(izip(files, output_files), CHUNK_SIZE):
        input_batch = map(first, batch)
        output_batch = map(second, batch)
        run_crfasrnn_in_batches(input_batch, output_batch, options.gpu)


def parse_options():
    parser = argparse.ArgumentParser(description='process segmentation')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    parser.add_argument('input_dir', metavar='OUTPUT_DIR', help='folder to process')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR', help='output directory.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()
    main(options)
