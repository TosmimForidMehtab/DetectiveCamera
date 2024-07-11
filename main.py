import cv2
import pygame
import threading
import numpy as np
from camera.camera_service import capture_frame
from utils.image_processing import enhance_low_light
from email_alert.email_service import EmailAlertService
from config.config import CAMERA_ID

net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/deploy.caffemodel")

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

motion_detected = False
alarm_playing = False


def initialize_sound():
    pygame.mixer.init()


def finalize_sound():
    pygame.mixer.quit()


def detect_human(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                return True
    return False


def process_frame(frame, previous_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame is None:
        return gray, None

    frame_delta = cv2.absdiff(previous_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return gray, contours


def handle_motion_detection(frame, contours):
    global motion_detected
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        motion_roi = frame[y : y + h, x : x + w]
        if detect_human(motion_roi):
            motion_detected = True
            break


def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def main():
    global motion_detected, alarm_playing
    previous_frame = None

    initialize_sound()

    try:
        while True:
            frame = capture_frame(CAMERA_ID)
            if frame is None:
                continue

            frame = enhance_low_light(frame)
            gray, contours = process_frame(frame, previous_frame)

            if contours is not None:
                handle_motion_detection(frame, contours)

            cv2.imshow("Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if motion_detected and not alarm_playing:
                play_sound("resources/sound.wav")
                alarm_playing = True

                image_path = "motion_detected.jpg"
                cv2.imwrite(image_path, frame)
                email_thread = EmailAlertService(image_path)
                email_thread.start()

                motion_detected = False

            if not motion_detected:
                alarm_playing = False

            if key == ord("q"):
                break

            previous_frame = gray

    finally:
        finalize_sound()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
