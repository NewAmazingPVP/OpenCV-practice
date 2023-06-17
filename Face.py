import asone
from asone import utils
from asone import ASOne
import cv2
import time

detector = ASOne(detector=asone.YOLOV8L_PYTORCH , use_cuda=True) # Set use_cuda to False for cpu
# for list https://github.com/augmentedstartups/AS-One/blob/main/asone/linux/Instructions/Benchmarking.md
# best peformance to quality ratio: YOLOV6S_REPOPT_PYTORCH

cap = cv2.VideoCapture(0)

# Set the camera's resolution
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create a named window and set it to fullscreen
window_name = "result"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    start_time = time.time()
    _, frame = cap.read()
    if not _:
        break

    dets, img_info = detector.detect(frame)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    # Calculate FPS and display it on the frame
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
