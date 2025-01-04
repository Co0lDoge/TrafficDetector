import cv2
import numpy as np
from ultralytics import YOLO, solutions
from collections import defaultdict
import os
import time
import pandas as pd
import tomllib

from args_loader import load_args
from sector import Sector
from regions_counter import RegionsCounter
from step_timer import StepTimer

def get_fps(cap) -> float|int:
    major_ver, _, _ = cv2.__version__.split('.')
    if int(major_ver) >= 3:
        return cap.get(cv2.CAP_PROP_FPS)
    return cap.get(cv2.cv.CV_CAP_PROP_FPS)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

video_path, model_path, output_path, report_path, regions = load_args()

# Подгружаем данные из TOML файла.
# Надеюсь Гуидо ван Россум простит меня за это.
with open("settings.toml", "rb") as f:
    settings = tomllib.load(f)

# Открываем видео
cap = cv2.VideoCapture(settings["video-path"])

frame_dt = 1/get_fps(cap)    # TODO подтягивать шаг кадра из файла
timer = StepTimer(frame_dt)

sector = Sector(
    settings["sector-length"],
    settings["lane-count"],
    settings["vehicle-classes"],
    timer,
    settings["observation-time"],
    settings["vehicle-size-coeffs"],
)
counter = RegionsCounter(settings["model-path"], regions=regions)

width, height = settings["target-width"], settings["target-height"]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter("output/order-479.mp4", fourcc, get_fps(cap), (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    counter.count(frame, annotate=True)
    sector.update(counter.regions["start"], counter.regions["end"])

    cv2.putText(
        frame,
        f"{sector.classwise_traveled_count}",
        (10, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255)
    )
    cv2.putText(
        frame,
        f"Current period timer: {int(sector.period_timer.time)}",
        (10, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255)
    )

    # Показ текущего кадра
    cv2.imshow("Crossroad Monitoring", frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        sector.new_period()
        break

stats = sector.traffic_stats()
print(stats)
stats.to_excel("output/traffic-stats.xlsx")

# Освобождаем ресурсы
cap.release()
output.release()
cv2.destroyAllWindows()
