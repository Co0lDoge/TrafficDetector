import cv2
import logging

def get_fps(cap) -> float|int:
    major_ver, _, _ = cv2.__version__.split('.')
    if int(major_ver) >= 3:
        return cap.get(cv2.CAP_PROP_FPS)
    return cap.get(cv2.cv.CV_CAP_PROP_FPS)

def open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Не удалось открыть видеофайл {video_path}")
        quit()
    else:
        logging.info(f"Видеофайл открыт успешно: {video_path}")
        fps = get_fps(cap)
        if fps > 0:
            logging.info(f"Частота кадров видеофайла: {fps:.2f} FPS")
        else:
            logging.warning("Частота кадров не может быть определена.")

    return cap, fps