import keras
from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.models.unet import vgg_unet

input_height_val = 480
input_width_val = 360
n_classes_val = 51

model = vgg_unet(n_classes=233 ,  input_height=320, input_width=480)
#img_input = keras.layers.Input(shape=(input_height_val,input_width_val , 3 ))

# Defining the encode layers for the network
#conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
#conv1 = keras.layers.Dropout(0.2)(conv1)
#conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

#conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#conv2 = keras.layers.Dropout(0.2)(conv2)
#conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)


#Defining the decode layers
#conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#conv3 = keras.layers.Dropout(0.2)(conv3)
#conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

#up1 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv3), conv2], axis=-1)
#conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
#conv4 = keras.layers.Dropout(0.2)(conv4)
#conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

#up2 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
#conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
#conv5 = keras.layers.Dropout(0.2)(conv5)
#conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

#out = keras.layers.Conv2D( n_classes_val, (1, 1) , padding='same')(conv5)
#model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model

model.train(
train_images = "bdd100k/seg/images/train/",
train_annotations = "bdd100k/seg/color_labels/train/",
checkpoints_path = "Checkpoints/",
epochs=20)

