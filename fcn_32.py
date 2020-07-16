import keras
import os
import h5py
from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models import fcn


model = fcn.fcn_32(n_classes=233 ,  input_height=200, input_width=200)
model.train(
train_images = "/content/drive/My Drive/bdd100k/seg/images/val",
train_annotations = "/content/drive/My Drive/bdd100k/seg/ne/val_c",
checkpoints_path = "./drive/My Drive/bdd100k_fcn2/ckptfin/" , epochs=20
 )
