import numpy as np
import cv2
from keras.models import model_from_json

class_names = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'BA',
    11: 'PA',
    12: 'CA'
}
no_of_classes = 13


def process_image(image):
    # grayscale conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply OTSU thresholding
    _, thresh1 = cv2.threshold(
        image, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # median blur
    image = cv2.medianBlur(thresh1, 3)

    # resizing for le-net we need (32,32) image shape
    image = cv2.resize(image, (32, 32))

    return image


json_file = open('model_lenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# loads weights into this model
model.load_weights("model_lenet.h5")
print("Model has been loaded")


def classify_character(image):
    # return preprocessed image of shape 32 X 32
    processed_image = process_image(image)
    predict_x = model.predict(processed_image.reshape(1, 32, 32, 1))
    # predict_x has 13 columns and each value in row represents the probability of a data that belongs to respected classes
    # print(predict_x.shape)
    class_of_x = np.argmax(predict_x, axis=1)
    # np.argmax with axis = 1, gives the column value having maximum probability
    # print(class_of_x)
    class_id = class_of_x[0]
    return class_id


image = cv2.imread("ba.png")
class_id = classify_character(image)
print(f"Class ID: {class_id}")
print(f"Detected character is : {class_names[class_id]}")
