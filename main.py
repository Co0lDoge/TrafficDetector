import cv2
import numpy as np
from ultralytics import YOLO, solutions
from collections import defaultdict
import os
import time
import pandas as pd
import tomllib
import argparse
import json

from sector import Sector
from regions_counter import RegionsCounter
from step_timer import StepTimer

# Добавление аргументов запуска
parser = argparse.ArgumentParser()
parser.add_argument("--video-path", type=str, required=True, help="Путь к видео")
parser.add_argument("--model-path", type=str, required=True, help="Путь к модельке")
parser.add_argument("--output-path", type=str, required=True, help="Путь для выходного файлы")
parser.add_argument("--report-path", type=str, required=True, help="Путь для выходного отчета")
parser.add_argument("--regions", type=str, required=True, help="Массив точек областей")

# Получение всех аргументов
args = parser.parse_args()

# Парсинг регионов
if args.regions:
    with open(args.regions, 'r') as f:
        regions = json.load(f)
else:
    regions = json.loads(args.regions)
print(f"Regions: {regions}")

# Доступ к аргументам
video_path = args.video_path
model_path = args.model_path
output_path = args.output_path
report_path = args.report_path

def get_fps(cap) -> float|int:
    major_ver, _, _ = cv2.__version__.split('.')
    if int(major_ver) >= 3:
        return cap.get(cv2.CAP_PROP_FPS)
    return cap.get(cv2.cv.CV_CAP_PROP_FPS)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
counter = RegionsCounter(settings["model-path"], settings["regions"])

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
