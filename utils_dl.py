import warnings
import cv2
from imutils import face_utils
import dlib
from imutils.video import VideoStream, FPS
import imutils
import numpy as np
import time
import math
import sqlite3
import pandas as pd
import os
# from __main__ import *
import pickle
import shutil
from os import listdir
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition as fr
from flask import jsonify
import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from skimage.io import imread
from skimage.filters import threshold_otsu, threshold_niblack, threshold_yen
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
from skimage.transform import resize
import joblib
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import backend as k
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from matplotlib.figure import Figure
from tensorflow.keras import layers
import PIL
from PIL import ImageOps
from matplotlib import cm
from tensorflow.keras.preprocessing.image import load_img
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.preprocessing.image import img_to_array

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
graph = tf.get_default_graph()
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'data/models/dl/anplr_gray_34ch.h5')
model = models.load_model(model_dir)


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

num_classes = 3
model_dir2 = os.path.join(current_dir, 'data/models/dl/char_seg.h5')
model2 = get_model((112, 208), num_classes)
model2.load_weights(model_dir2)
model_dir3 = os.path.join(current_dir, 'data/models/dl/loc_seg.h5')
model3 = get_model((368, 640), num_classes)
model3.load_weights(model_dir3)
with open('dict_ocr_34.json', 'r') as f:
    labels = json.load(f)
letters = list(labels)

black_list = ["GMK8135", "GF66701", "GF66712", "04134"]

sqlite3.register_adapter(np.int64, lambda val: int(val))
shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
train_dir = "data/images/FaVe/snap"
def dilate_transformation(img):
    binary_car_image = img

    th1 = cv2.dilate(binary_car_image, np.ones((7, 1), np.uint8), iterations=2)

    th2 = cv2.dilate(binary_car_image, np.ones((8, 1), np.uint8), iterations=2)

    th3 = cv2.dilate(binary_car_image, np.ones((9, 1), np.uint8), iterations=2)

    th4 =cv2.dilate(binary_car_image, np.ones((10, 1), np.uint8), iterations=2)

    th5 = cv2.dilate(binary_car_image, np.ones((11, 1), np.uint8), iterations=2)
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(th1)
    plt.title('k 1')
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(th2)
    plt.title('k 2')
    fig3, ax3 = plt.subplots(1)
    ax3.imshow(th3)
    plt.title('k 3')
    fig4, ax4 = plt.subplots(1)
    ax4.imshow(th4)
    plt.title('k 4')
    fig5, ax5 = plt.subplots(1)
    ax5.imshow(th5)
    plt.title('k 5 ')
    plt.show()
def erode_transformation(img):
    binary_car_image = img

    th1 = cv2.erode(binary_car_image, np.ones((7, 1), np.uint8), iterations=2)

    th2 = cv2.erode(binary_car_image, np.ones((8, 1), np.uint8), iterations=2)

    th3 = cv2.erode(binary_car_image, np.ones((9, 1), np.uint8), iterations=2)

    th4 =cv2.erode(binary_car_image, np.ones((10, 1), np.uint8), iterations=2)

    th5 = cv2.erode(binary_car_image, np.ones((11, 1), np.uint8), iterations=2)
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(th1)
    plt.title('k 1')
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(th2)
    plt.title('k 2')
    fig3, ax3 = plt.subplots(1)
    ax3.imshow(th3)
    plt.title('k 3')
    fig4, ax4 = plt.subplots(1)
    ax4.imshow(th4)
    plt.title('k 4')
    fig5, ax5 = plt.subplots(1)
    ax5.imshow(th5)
    plt.title('k 5 ')
    plt.show()
def open_transformation(img):
    binary_car_image = img
    th1 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
    th2 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    th3 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    th4 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    th5 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(th1)
    plt.title('k 1')
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(th2)
    plt.title('k 2')
    fig3, ax3 = plt.subplots(1)
    ax3.imshow(th3)
    plt.title('k 3')
    fig4, ax4 = plt.subplots(1)
    ax4.imshow(th4)
    plt.title('k 4')
    fig5, ax5 = plt.subplots(1)
    ax5.imshow(th5)
    plt.title('k 5 ')
    plt.show()
def close_transformation(img):
    binary_car_image = img
    th1 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    th2 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    th3 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
    th4 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    th5 = cv2.morphologyEx(np.float32(binary_car_image), cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(th1)
    plt.title('k 1')
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(th2)
    plt.title('k 2')
    fig3, ax3 = plt.subplots(1)
    ax3.imshow(th3)
    plt.title('k 3')
    fig4, ax4 = plt.subplots(1)
    ax4.imshow(th4)
    plt.title('k 4')
    fig5, ax5 = plt.subplots(1)
    ax5.imshow(th5)
    plt.title('k 5 ')
    plt.show()
def cv2_threshold_BINARY(img, ka):
    gray_car_image = img
    k0,k1,k2,k3,k4 = ka[0],ka[1],ka[2],ka[3],ka[4]
    ret1, th1 = cv2.threshold(gray_car_image, k0, 255, cv2.THRESH_BINARY)
    ret1, th2 = cv2.threshold(gray_car_image, k1, 255, cv2.THRESH_BINARY)
    ret1, th3 = cv2.threshold(gray_car_image, k2, 255, cv2.THRESH_BINARY)
    ret1, th4 = cv2.threshold(gray_car_image, k3, 255, cv2.THRESH_BINARY)
    ret1, th5 = cv2.threshold(gray_car_image, k4, 255, cv2.THRESH_BINARY)
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(th1)
    plt.title('k 1')
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(th2)
    plt.title('k 2')
    fig3, ax3 = plt.subplots(1)
    ax3.imshow(th3)
    plt.title('k 3')
    fig4, ax4 = plt.subplots(1)
    ax4.imshow(th4)
    plt.title('k 4')
    fig5, ax5 = plt.subplots(1)
    ax5.imshow(th5)
    plt.title('k 5 ')
    plt.show()
def filter_color(img, lower_red,upper_red):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    th1 = cv2.inRange(img, lower_red, upper_red)
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(th1)
    plt.title('k 1')
    plt.show()
#aqui comienza ranpv

def get_plate_coor(gray_image,rgb_image):
    gray_car_image = gray_image 
    #print(car_image)
    car_image = rgb_image.copy()
    with graph.as_default(), session.as_default():
        val_preds = model3.predict(car_image[tf.newaxis, ...])
    mask = np.argmax(val_preds[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = keras.preprocessing.image.array_to_img(mask)
    mask = img_to_array(mask)
    ret, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    #cv2_threshold_BINARY(mask, [10, 75, 125, 175, 240])
  #  mask  = cv2.morphologyEx(np.float32(mask), cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    #mask  = cv2.morphologyEx(np.float32(mask), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
   # close_transformation(np.float32(mask))
    #mask  = cv2.morphologyEx(np.float32(mask), cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    
    #fig2, ax2 = plt.subplots(1)
   # ax2.imshow(mask)
    label_image = measure.label(mask, background=1, connectivity=2)
    plate_dimensions = (
        0.02 * label_image.shape[0], 0.9 * label_image.shape[0], 0.01 * label_image.shape[1],
        0.9 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    license_plate_mask = car_image
    cv2.rectangle(license_plate_mask, (0, 0), (license_plate_mask.shape[1], license_plate_mask.shape[0]),
                  (127, 0, 0), -1)
    # print("aqui comienza regionprops")
    for region in regionprops(label_image):
        if region.area < 1400:
            continue
        if region.area > 24000:
            continue
        min_row, min_col, max_row, max_col = region.bbox[0]+3, region.bbox[1], region.bbox[2]-3, region.bbox[3]-2
        x0, y0,x1, y1 =  min_col, min_row, max_col, max_row
        region_height = max_row - min_row
        region_width = max_col - min_col
       # if region_width < 1.4 * region_height:
          #  continue
      #  if region_width > 2.8 * region_height:
       #     continue
        if y0 < 10:
            continue
        if x0 <10:
            continue
       # if x1 > label_image.shape[1]-10:
          #  continue
        if y1 > label_image.shape[0]-20:
            continue
        # print(region.bbox)
        if min_height <= region_height <= max_height and min_width <= region_width <= max_width and \
                region_width > region_height:
            # print(region.area)
            if min_row != 0 and min_row != 0 and max_row != 0 and max_col != 0:
                plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
                cv2.rectangle(license_plate_mask, (x0, y0), (x1, y1), (0, 0, 127), -1)
                cv2.rectangle(license_plate_mask, (x0, y0), (x1, y1), (0, 127, 0), 2)
               # fig, ax1 = plt.subplots(1)
               # ax1.imshow(gray_car_image[min_row:max_row, min_col:max_col], cmap="gray")
    files_m = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/plate_loc/') if "masks" in f]
    files_mix = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/plate_loc/') if "mix" in f]
    files_o = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/plate_loc/') if "original" in f]
    counter_o = len(files_o)
    counter_m = len(files_m)
    counter_mix = len(files_mix)
    direction_o = current_dir + '/data/images/ANPR/data_gen/plate_loc/' + "original" + '_%s.jpg' % counter_o
    direction_m = current_dir + '/data/images/ANPR/data_gen/plate_loc/' + "masks" + '_%s.jpg' % counter_m
    direction_mix = current_dir + '/data/images/ANPR/data_gen/plate_loc/' + "mix" + '_%s.jpg' % counter_mix

    logo = license_plate_mask
    room = rgb_image
    nah, logo_mask = cv2.threshold(logo[:, :, 0], 20, 255, cv2.THRESH_BINARY)
    logo_mask = abs(logo_mask - 255)
    room2 = room.copy()
    room2[np.where(logo_mask == 0)] = logo[np.where(logo_mask == 0)]
    #plt.show()
    cv2.imwrite(direction_o, room)
    cv2.imwrite(direction_m, license_plate_mask)
    cv2.imwrite(direction_mix, room2)
    # print(plate_objects_cordinates)
    return plate_objects_cordinates

def plate_segmentation(plate_like_objects,plate_like_objects2):
    chars = []
    col = []
    for i in range(len(plate_like_objects)):
        license_plate_rgb = resize(plate_like_objects2[i], (112, 208))
        license_plate_o = resize(plate_like_objects[i], (112, 208))
        license_plate = license_plate_o.copy()
        global model2, graph, session
        license_plate_to_seg = license_plate_rgb
        license_plate_to_seg = license_plate_to_seg*255
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        filter_color(np.float32(license_plate_to_seg), np.array([0, 145, 0]),np.array([255, 218, 255]))
        #hsv = cv2.cvtColor(np.float32(license_plate_to_seg), cv2.COLOR_GRB2HSV)
        lower_red = np.array([80, 0, 0])
        upper_red = np.array([200, 255, 255])
        th1 = cv2.inRange(np.float32(license_plate_rgb), lower_red, upper_red)
        fig5, ax5 = plt.subplots(1)
        ax5.imshow(th1)
        with graph.as_default(), session.as_default():
            val_preds = model2.predict(license_plate_to_seg[tf.newaxis, ...])
        mask = np.argmax(val_preds[0], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask = keras.preprocessing.image.array_to_img(mask)
        mask = img_to_array(mask)
	
        ret, th1 = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        th1 = cv2.dilate(np.float32(th1), np.ones((15, 1), np.uint8), iterations=1)
        th1 = cv2.erode(np.float32(th1), np.ones((15, 1), np.uint8), iterations=1)
        mask = th1
       # fig, ax1 = plt.subplots(1)
       # ax1.imshow(mask)
       
        labelled_plate = measure.label(mask, background=1, connectivity=2)
        fig, ax1 = plt.subplots(1)
        ax1.imshow(mask, cmap="gray")
        character_dimensions = (
            0.25 * license_plate.shape[0], 0.7 * license_plate.shape[0], 0.02 * license_plate.shape[1],
            0.13 * license_plate.shape[1])
        min_height, max_height, min_width, max_width = character_dimensions
        characters = []
        column_list = []
        centroids = []
        license_plate_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        #print(license_plate_mask.shape)
        cv2.rectangle(license_plate_mask, (0, 0), (license_plate_mask.shape[1], license_plate_mask.shape[0]),
                      (127, 0, 0), -1)
        for regions in sorted(regionprops(labelled_plate), key=lambda r: r.area, reverse=True,):

            if regions.area < 200:
                continue
            if regions.area > 10000:
                continue
            # print(regions.area)
            y0, x0, y1, x1 = regions.bbox[0] , regions.bbox[1] , regions.bbox[2] , regions.bbox[3]
            region_height, region_width = y1 - y0, x1 - x0
            #if region_width < .18 * region_height:
               # continue
            #if region_width > 0.3 * region_height:
               # continue
            if x1 > license_plate.shape[1] - 1:
                continue
            if y0 < 1:
                continue
            if y1 > license_plate.shape[0] - 1:
                continue
            if x0 < 1:
                continue
            if min_height < region_height < max_height and min_width < region_width < max_width:
                if len(centroids) == 0:
                    centroids.append(regions.centroid[1])
                    roi = license_plate[y0:y1, x0:x1]
                    rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
                    ax1.add_patch(rect_border)
                    if roi.shape[0] != 0 and roi.shape[1] != 0:
                        resized_char = resize(roi, (20, 40))
                        characters.append(resized_char)
                        column_list.append(x0)

                        cv2.rectangle(license_plate_mask, (x0, y0), (x1, y1), (0,0,127), -1)
                        cv2.rectangle(license_plate_mask, (x0 , y0 ), (x1, y1 ), (0, 127, 0), 2)
                        files = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/letters_from_videos/')]
                        counter = len(files)
                        direction = current_dir + '/data/images/ANPR/data_gen/letters_from_videos/' + "letter" + '_%s.jpg' % counter
                        cv2.imwrite(direction, (roi* 255).astype(np.uint8) )

                if len(centroids) != 0:
                    distances = []
                    for centroid in centroids:
                        distance = abs(centroid - regions.centroid[1])
                        distances.append(distance)
                    if all(i >= 0.5 * region_width for i in distances):
                        roi = license_plate[y0:y1, x0:x1]
                        if roi.shape[0] != 0 and roi.shape[1] != 0:
                            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
                            ax1.add_patch(rect_border)
                            resized_char = resize(roi, (20, 40))
                            characters.append(resized_char)
                            column_list.append(x0)
                            #print(license_plate_mask)

                            cv2.rectangle(license_plate_mask, (x0, y0), (x1, y1), (0,0,127), -1)
                            cv2.rectangle(license_plate_mask, (x0, y0), (x1, y1), (0, 127, 0), 2)
                            files = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/letters_from_videos/')]
                            counter = len(files)
                            direction = current_dir + '/data/images/ANPR/data_gen/letters_from_videos/' + "letter" + '_%s.jpg' % counter
                            cv2.imwrite(direction, (roi* 255).astype(np.uint8) )
                    centroids.append(regions.centroid[1])
        files_o = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/masks/') if "original" in f]
        files_m = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/masks/') if "masks" in f]
        files_mix = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/masks/') if "mix" in f]
        counter_o = len(files_o)
        counter_m = len(files_m)
        counter_mix = len(files_mix)
        direction_o = current_dir + '/data/images/ANPR/data_gen/masks/' + "original" + '_%s.jpg' % counter_o
        direction_m = current_dir + '/data/images/ANPR/data_gen/masks/' + "masks" + '_%s.jpg' % counter_m
        direction_mix = current_dir + '/data/images/ANPR/data_gen/masks/' + "mix" + '_%s.jpg' % counter_mix
        logo = license_plate_mask
        room = license_plate_rgb
        nah, logo_mask = cv2.threshold(logo[:, :, 0], 20, 255, cv2.THRESH_BINARY)
        logo_mask = abs(logo_mask - 255)
        room2 = room.copy()
        room2[np.where(logo_mask == 0)] = logo[np.where(logo_mask == 0)]
        print(len(logo), room.shape)
        fig2, ax2 = plt.subplots(1)
        ax2.imshow((room2* 255).astype(np.uint8))
        col.append(column_list)
        chars.append(characters)
        plt.show()
        #print(len(chars[0]))
        if len(chars[0]) ==5:
            cv2.imwrite(direction_o, (license_plate_rgb* 255).astype(np.uint8))
            cv2.imwrite(direction_m, license_plate_mask)
            cv2.imwrite(direction_mix, (room2* 255).astype(np.uint8))
    return chars, col


def plate_prediction(chars_list, col_index):
    plates_numbers = []
    for each_str, each_col in zip(chars_list, col_index):
        classification_result = []
        if len(each_str) <= 3:
            rightplate_string = "No_plate"
        if len(each_str) >= 4:
            for each_character in each_str:
                global model, graph, session
                #each_character = np.expand_dims(each_character, axis=0)
                each_character = each_character.reshape(1, 20, 40, 1).astype('float')
                with graph.as_default(), session.as_default():
                    result = model.predict(each_character, steps=1, batch_size=128)
                result2 = [str(letters[np.where(result == np.amax(result))[1][0]])]
                classification_result.append(result2)
            plate_string = ''
            for eachPredict in classification_result:
                plate_string += eachPredict[0]
            column_list_copy = each_col[:]
            each_col.sort()
            rightplate_string = ''
            for each in each_col:
                rightplate_string += plate_string[column_list_copy.index(each)]
                print(rightplate_string)
            rightplate_string = rightplate_string.replace('Q', '')
        plates_numbers.append(rightplate_string)
    return plates_numbers
def gen2():
    t1 = time.time()
    video_path = "data/vids/06159/06159_0.mp4"
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if ret:
            frame_small, frame_gray = cv2.resize(frame, (640, 368)), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            coordinates = get_plate_coor(frame_gray_small, frame_small)
            number_and_plate = []
            for cordinate in coordinates:
                y_min, x_min, y_max, x_max = cordinate[0] * 3, cordinate[1] * 3, cordinate[2] * 3, cordinate[3] * 3
                y_min2, x_min2, y_max2, x_max2 = cordinate[0], cordinate[1], cordinate[2], cordinate[3]
                if y_max2 > 345:
                    continue
                if y_min2 < 15:
                    continue
                if x_max2 > 625:
                    continue
                if x_min2 < 15:
                    continue
                chars, cols = plate_segmentation([frame_gray[y_min:y_max, x_min:x_max]],[frame[y_min:y_max, x_min:x_max]])
                plate_numbers = plate_prediction(chars, cols)
                print(plate_numbers)

                for plate_number in plate_numbers:
                    if plate_number is not None:
                        number_and_plate.append([plate_number, y_min2, x_min2, y_max2, x_max2])

                        if plate_number is not "No_plate" and len(plate_number) > 3:

                            plate_to_save = frame[y_min:y_max, x_min:x_max]
                            #cv2.putText(plate_to_save, plate_number, (30, 30),
                                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            files = [f for f in listdir(current_dir + '/data/images/ANPR/data_gen/recorded_plates/')]
                            counters = len(files)
                            direction = current_dir + '/data/images/ANPR/data_gen/recorded_plates/' + plate_number + '_%s.jpg' % counters
                            cv2.imwrite(direction, plate_to_save)

            for data_row in number_and_plate:
                plate_number, y_min2, x_min2, y_max2, x_max2 = data_row[0], data_row[1], data_row[2], data_row[3], \
                                                               data_row[4]
                if plate_number is not "No_plate":
                    cv2.putText(frame_small, plate_number, (x_min2 - 6, y_min2 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame_small, (x_min2, y_min2), (x_max2, y_max2),
                                  (0, 0, 255), 2)
                    for plate_s in black_list:
                        if fuzz.ratio(plate_s, plate_number) > 89 and fuzz.ratio(plate_s, plate_number) < 100:
                            cv2.putText(frame_small, plate_s, (x_max2 + 6, y_max2 + 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if fuzz.ratio(plate_s, plate_number) > 99:
                            cv2.putText(frame_small, plate_s, (x_max2 + 6, y_max2 + 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            (flag, encodedImage) = cv2.imencode(".jpg", frame_small)
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
        else:
            # aqui deberiamos intentar editar frame_small para que agrege a la derecha las palcas encontradas de la
            # lista negra
            folder = current_dir + '/data/images/ANPR/data_gen/recorded_plates/'
            files = [f for f in listdir(folder)]
            files2 = []

            for file in files:
                file2 = file.split("_", 1)
                files2.append(file2[0])

            keys, counts = np.unique(files2, return_counts=True)

            x = 0.2*max(counts)


            for file in keys[np.where(counts <= x)]:
                files2 = [y for y in files2 if y != file]
            keys, counts = np.unique(files2, return_counts=True)



            for key1 in keys:
                #print("for", key1)
                for key2 in keys:
                    rati = fuzz.ratio(key1,key2)
                    if 70 < rati < 100:
                        #print(key1,counts[keys==key1], key2,counts[keys==key2], rati)
                        if counts[keys==key1] > counts[keys==key2]:
                            files2 = [y for y in files2 if y != key2]

            keys, counts = np.unique(files2, return_counts=True)
            plt.bar(keys, counts)
            plt.show()
            lastplot = len(keys)
            fig = plt.figure()
            k = 1
            for key in keys:
                last_list = [f for f in files if f.split("_", 1)[0]==key]
                print(last_list[1])
                img = plt.imread(folder+last_list[1])
                ax1 = fig.add_subplot(lastplot, 1, k)
                ax1.imshow(img)
                k +=1
            plt.show()

            for filename in listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            break
    video_capture.release()
    t2 = time.time()
    print(t2 - t1)


#esto es ranpv
# esto es FaVe
def functionist():
    x = []
    y = []
    verbose = True
    name_list = [classes for classes in os.listdir(train_dir)]
    for class_dir in name_list:
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = fr.load_image_file(img_path)
            face_bounding_boxes = fr.face_locations(image)
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"), end="\r")
            else:
                x.append(fr.face_encodings(image, known_face_locations=face_bounding_boxes, num_jitters=10)[0])
                y.append(class_dir)
    dictionary = dict(zip(y, x))
    # print("encodes creados")
    return dictionary


def face_rec(file, encodes):
    encodes = encodes.drop(columns=['index'])
    encodes["folio"] = encodes["folio"].astype(str)
    face_names = list(encodes["nombre"])
    folios = list(encodes["folio"])
    encodes = encodes.drop(columns=['nombre'])
    encodes = encodes.drop(columns=['folio'])
    face_encodings = np.array([encodes.iloc[number] for number in range(0, len(encodes))])
    unknown_image = fr.load_image_file(file)
    unknown_encoding = fr.face_encodings(unknown_image)[0]
    results = fr.compare_faces(face_encodings, unknown_encoding, tolerance=0.45)
    tole = fr.face_distance(face_encodings, unknown_encoding)
    resuls_str = []
    for result in results:
        resuls_str.append(str(result))
    dict_res1 = {"personas": [{"nombre": k, "folio": l, "distance": d, "result": v} for k, l, d, v in
                              zip(face_names, folios, tole, resuls_str)]}
    valores = pd.DataFrame(dict_res1["personas"])
    valores = valores.sort_values(by=['distance'])
    dict_res = valores[valores.result != "False"]
    x = dict_res["nombre"]
    y = dict_res["folio"]
    z = dict_res["distance"]
    dict_res = {"personas": [{"folio": l, "nombre": k, "distance": m} for k, l, m in zip(x, y, z)]}
    return dict_res


def encode_creation(encode, id_num, params):
    # print("inicio de creacion de encodes")
    nombre, folio = params["nombre"], params["folio"]
    biden_values = encode
    biden_values = pd.DataFrame(biden_values.values())
    biden_values.insert(loc=0, column="nombre", value=nombre)
    biden_values.insert(loc=0, column="folio", value=folio)
    biden_values.insert(loc=0, column="index", value=id_num)
    for count, filename in enumerate(os.listdir("data/images/FaVe/snap/1")):
        dst = folio + "_" + str(count) + ".jpg"
        src = 'data/images/FaVe/snap/1/' + filename
        dst = 'data/images/FaVe/snap/1/' + dst
        os.rename(src, dst)
    files = [file for file in listdir("data/images/FaVe/snap" + "/1/")]
    for file in files:
        shutil.move('data/images/FaVe/snap/1/' + file, '/data/images/FaVe/enrolleds')
    # print("fotografias guardadas, enrolamiento completo")
    return biden_values
#aqui termina FaVe

def gen(encos):
    t1 = time.time()
    total = 0
    is_known = 0
    name_to_return = "noname"
    video_capture = cv2.VideoCapture(0)
    known_face_encodings = encos
    known_face_names, known_folios = known_face_encodings["nombre"], known_face_encodings["folio"]
    dict_folios = dict(zip(known_face_names, known_folios))
    known_face_encodings = known_face_encodings.drop(columns=['index'])
    known_face_encodings = known_face_encodings.drop(columns=['folio'])
    known_face_encodings = known_face_encodings.drop(columns=['nombre'])
    face_locations = []

    face_names = []
    process_this_frame = True
    while True:
        noninteresting, frame = video_capture.read()
        rgb_small_frame = frame[:, :, ::-1]

        if process_this_frame:
            face_locations = fr.face_locations(rgb_small_frame)

            # if len(face_locations) > 0:
            # print("face_locations", face_locations)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                name = "Unknown"
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
        # cv2.rectangle(frame, (100, 30), (280, 250), (0, 0, 250), 8)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if name != "Unknown":
                is_known += 1
            if is_known > 10:
                name_to_return = name
                break
            if name == "Unknown":
                folio = "10000001"
            else:
                folio = dict_folios[name]
            cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 15), (0, 0, 255), 2)
            cv2.rectangle(frame, (left - 10, bottom + 15), (right + 10, bottom + 55), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, folio, (left + 6, bottom + 35), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, name, (left + 6, bottom + 50), font, 0.5, (255, 255, 255), 1)
            if is_known > 7:
                cv2.putText(frame, "sujeto identificado como {}".format(name), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)
                cv2.putText(frame, "espere a que el video se detenga y precione el boton", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if total > 7:
                cv2.putText(frame, "Identidad capturada, espere a que el video se detenga y precione el boton",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            total += 1
            time.sleep(.1)
            nonimportance, frame2 = video_capture.read()
            frame3 = frame2.copy()
            img_name = "/data/images/FaVe/snap/1/OpenCV_frame_{}.jpg".format(total)
            # print("writhing", img_name)
            cv2.imwrite(img_name, frame3)
        if name_to_return != "noname":
            video_capture.release()
            # myobj = {'folio': folio, "nombre":name_to_return}
            # x = requests.post('http://0.0.0.0:5000/id_cap', data=myobj)
            with open('data/name.dat', 'wb') as file:
                pickle.dump(name_to_return, file)
            t2 = time.time()
            print(t2 - t1)
            break
        if total == 10:
            with open('data/name.dat', 'wb') as file:
                pickle.dump(name_to_return, file)
            # myobj = {'folio': folio, "nombre":name_to_return}
            # x = requests.post('http://0.0.0.0:5000/id_cap', data=myobj)
            video_capture.release()
            t2 = time.time()
            print(t2 - t1)
            break
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


