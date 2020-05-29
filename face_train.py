import os
from PIL import Image
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    "C:/Users/SHREYANK KINI/datascience/face-mask-detector/cdata/haarcascade_frontalface_alt2.xml"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imgage_dir = os.path.join(BASE_DIR, "images")

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(imgage_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5
            )

            for (x, y, w, h) in faces:
                roi = image_array[y : y + h, x : x + h]
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
