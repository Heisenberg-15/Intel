from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
import tensorflow as tf
model = pspnet_50_ADE_20K()
#model.trainable = False
print(model.__code__.co_varnames)
model = tf.keras.Sequential([
    model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(233, activation='softmax')
])
model.train(
train_images = "./drive/My Drive/bdd100k/seg/images/train/",
train_annotations = "./drive/My Drive/bdd100k/seg/ne/train/",
checkpoints_path = "./drive/My Drive/bdd_psp/ckpt/" , epochs=10
)
out = model.predict_segmentation(
    inp="./drive/My Drive/bdd100k/seg/images/train/00d1c9e3-a7a7075f.jpg",
    out_fname="./drive/My Drive/output32.png"
)
#resnet
"""from keras.applications import InceptionResNetV2

from keras import layers
from keras import models
conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(300,300,3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(300*300, activation='softmax'))
model.train(
train_images = "./drive/My Drive/bdd100k/seg/images/train/",
train_annotations = "./drive/My Drive/bdd100k/seg/ne/train/",
checkpoints_path = "./drive/My Drive/bdd_psp/ckpt/" , epochs=10

)"""
