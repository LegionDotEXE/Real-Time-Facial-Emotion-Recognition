# LEGIONDOTEXE - GITHUB
# INITIAL CODE - REAL TIME FACIAL EMOTION DETECTION
# TENSORFLOW framework requires graphics card on the system - preferrably GTX 650 above or equivalent 

import cv2
import numpy as np
from keras.models import model_from_json

# This real-time facial emotion tool has been trained on the pre-existing facial expression dataset {More info on README.md}

# Pre-trained model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# Face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define a function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Open the webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()      #frames from webcam
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Color gradient switched to Gryascale

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    try:
        for (x, y, w, h) in faces:
            # Extract the face region
            face_image = gray[y:y + h, x:x + w]

            # rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_image = cv2.resize(face_image, (48, 48))   # Resized to model's input size (48x48)

            # Extract features and make a prediction
            img_features = extract_features(face_image)
            pred = model.predict(img_features)
            prediction_label = labels[pred.argmax()]

            # Display the emotion label near the face
            cv2.putText(frame, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))


        cv2.imshow("Emotion Detection", frame)  # Display the frame with emotions detected


        if cv2.waitKey(1) == 27:         # Exit the loop when the 'ESC' key is pressed
            break
    except cv2.error:
        pass

webcam.release()            # Webcam Release
cv2.destroyAllWindows()
