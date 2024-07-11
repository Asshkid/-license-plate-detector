from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# загружаем модель
coco_model = YOLO('yolov8n.pt')# детектим  машины
license_plate_detector = YOLO('license_plate_detector.pt') # обнаруживаем таблицу с номером

# загружаем видео
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7] # детектим тоолько те обьекты которые нам нужны со всей базы


frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # детектим транспортное средство
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # отслеживаем 
        track_ids = mot_tracker.update(np.asarray(detections_))

        # детектим номерной знак
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # привязываем номерной знак к машине
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # вырезаем номерной знак
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

               
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # читаем номерной знак
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# записываем результаты
write_csv(results, './test.csv')
