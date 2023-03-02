#importing the required libraries
import cv2
import face_recognition

# Load images and corresponding names
image_paths = ["jennie.jpg", "lisa.jpg", "rose.jpg","jissoo.jpg"]
names = ["Jennie", "Lisa", "Rose","Jissoo"]

# Load and encode images
images = [face_recognition.load_image_file(path) for path in image_paths]
encodings = [face_recognition.face_encodings(image)[0] for image in images]

# Start video capture
#Note: enter the index value as 1 if the web camera is used else use 0 to use default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame to speed up face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Scale back up face locations since the frame was scaled down to speed up detection
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(encodings, face_encoding)

        # Find the index of the first match
        if True in matches:
            match_index = matches.index(True)

            # Display the name of the person
            name = names[match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
