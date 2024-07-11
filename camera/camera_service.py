import cv2


def capture_frame(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame
