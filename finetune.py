#coding: utf-8
import os
import sys
import glob
import argparse
# import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=2, allow_soft_placement=True,  device_count = {'CPU': 64})
session = tf.Session(config=config)
K.set_session(session)

os.environ["OMP_NUM_THREADS"] = "64"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS = 15
BAT_SIZE = 32
FC_SIZE = 512 #1024
NB_IV3_LAYERS_TO_FREEZE = 172 #total layers:314


def get_nb_files(directory):
    # """
    # Get number of files by searching directory recursively
    # """
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    # """
    # Freeze all layers and compile the model
    # """
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
             'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
             'pant_length_labels']
num = [5,10,6,8,5,8,5,6]
directory = '/home/u12292/test/base/Images/neckline_design_labels'
nb_classes = 10
batch_size = BAT_SIZE
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical'
)
base_model = InceptionV3(weights='imagenet', include_top=False)
model = add_new_last_layer(base_model, nb_classes)
# setup_to_transfer_learn(model, base_model)
nb_epoch = NB_EPOCHS
nb_train_samples = get_nb_files(directory)
# history_tl = model.fit_generator(
#     train_generator,
#     nb_epoch=nb_epoch,
#     samples_per_epoch=nb_train_samples,
#     # validation_data=validation_generator,
#     # nb_val_samples=nb_val_samples,
#     class_weight='auto')

setup_to_finetune(model)
history_ft = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    class_weight='auto')
model.save('inceptionv3_ft_neckline_design_labels1.model')