import cv2
import os
import tomllib
import pandas as pd

from args_loader import load_args, region_adapt
from sector import Sector
from regions_counter import RegionsCounter
from step_timer import StepTimer

def get_fps(cap) -> float|int:
    major_ver, _, _ = cv2.__version__.split('.')
    if int(major_ver) >= 3:
        return cap.get(cv2.CAP_PROP_FPS)
    return cap.get(cv2.cv.CV_CAP_PROP_FPS)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

video_path, model_path, output_path, report_path, list_region = load_args()

# Подгружаем данные из TOML файла.
# Надеюсь Гуидо ван Россум простит меня за это.
with open("settings.toml", "rb") as f:
    settings = tomllib.load(f)

# Открываем видео
cap = cv2.VideoCapture(video_path)

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
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
width, height = settings["target-width"], settings["target-height"]

regions = region_adapt(list_region, video_width, width)
counter = RegionsCounter(model_path, regions=regions)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, get_fps(cap), (width, height))

# Флаг для генерации последнего отчета
generate_report = True  
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
        generate_report = False
        sector.new_period()
        break

# Генерация отчета за последний период
# TODO: не генерировать, если нажато q
if generate_report:
    sector.new_period()

traffic_stats = sector.traffic_stats()
classwise_stats = sector.classwise_stats()
print(traffic_stats)
print(classwise_stats)

merged_stats = pd.concat([traffic_stats, classwise_stats], axis=1)
merged_stats.to_excel(report_path)

# Освобождаем ресурсы
cap.release()
output.release()
cv2.destroyAllWindows()
