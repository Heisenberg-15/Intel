import tensorflow as tf
from google.colab.patches import cv2_imshow
m = tf.keras.metrics.MeanIoU(num_classes=253)
import cv2
img1 = cv2.imread("/content/drive/My Drive/bdd100k/seg/images/val/8ab12dec-59848511.jpg")
img2 = cv2.imread("./drive/My Drive/output23.jpg")
print(img1.shape)
print(img2.shape)
#print(img2)
m.update_state(img1/255,img2/255)
m.result().numpy()
