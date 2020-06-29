from keras_segmentation.predict import predict
# import cv2

predict(checkpoints_path = "Checkpoints/",
        inp = "bdd100k/seg/images/train/00d1c9e3-a7a7075f.jpg",
        out_fname = "output.png"
        )
