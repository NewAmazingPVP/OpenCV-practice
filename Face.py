import math
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        focal_length = 600  # Focal length of the camera in pixels **NOTE: This is approximate**
        face_width = w  
        distance = (2 * focal_length * 20) / face_width  # (20 is the size of the face in centimeters)
        distance = round(distance / 100, 2)

        fov = 78 # in degrees
        angle = math.atan(((x + w/2) - frame.shape[1]/2) / (0.5 * frame.shape[1] / math.tan(0.5 * math.radians(fov))))
        angle = round(math.degrees(angle), 2)

        cv2.putText(frame, f"{angle} deg", (x+w-60, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{distance} m", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
