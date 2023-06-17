import asone
from asone import utils
from asone import ASOne
import cv2

detector = ASOne(detector=asone.YOLOV8L_PYTORCH , use_cuda=True) # Set use_cuda to False for cpu
# for list https://github.com/augmentedstartups/AS-One/blob/main/asone/linux/Instructions/Benchmarking.md
# best peformance to quality ratio: YOLOV6S_REPOPT_PYTORCH


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    if not _:
        break
    
    dets, img_info = detector.detect(frame)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    cv2.imshow('result', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break