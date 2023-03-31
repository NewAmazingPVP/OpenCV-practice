import math
import cv2

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Estimate the distance to the face
        focal_length = 600  # Focal length of the camera in pixels **NOTE: This is approximate**
        face_width = w  # Width of the face in pixels
        distance = (2 * focal_length * 20) / face_width  # Distance in centimeters (20 is the size of the face in centimeters)
        distance = round(distance / 100, 2)  # Convert to meters and round to two decimal places

        # Calculate the angle of the face relative to the camera
        fov = 78 # in degrees
        angle = math.atan(((x + w/2) - frame.shape[1]/2) / (0.5 * frame.shape[1] / math.tan(0.5 * math.radians(fov))))
        angle = round(math.degrees(angle), 2)

        # Display the distance and angle next to the face
        cv2.putText(frame, f"{angle} deg", (x+w-60, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{distance} m", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('frame', frame)
    
    cv2.waitKey(1)

    # Exit if the user presses the 'e' key
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
