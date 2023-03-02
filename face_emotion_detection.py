#importing the required libraries
import cv2
import numpy as np

# Load the pre-trained model
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'emotion_net.caffemodel')

# Load the image
image = cv2.imread('face.jpg')

# Resize the image to 300x300 pixels and preprocess it
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network to get the predictions
model.setInput(blob)
predictions = model.forward()

# Get the label of the emotion with the highest probability
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_label = emotions[predictions[0].argmax()]

# Draw the label on the image
cv2.putText(image, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image
cv2.imshow('Facial Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
