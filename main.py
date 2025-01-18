import cv2
import os
import tomllib
import pandas as pd
import logging

from args_loader import load_args, get_adapted_region_points
from sector import SectorCluster
from regions_counter import RegionsCounter
from step_timer import StepTimer

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

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
logging.info(f"Загруженные настройки: {settings}")

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
width, height = settings["target-width"], settings["target-height"]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_dt = 1/fps    # TODO подтягивать шаг кадра из файла
timer = StepTimer(frame_dt)
regions = get_adapted_region_points(list_region, video_width, width)
counter = RegionsCounter(model_path, regions_points=regions)

sector = SectorCluster(
    settings["sector-length"],
    settings["lane-count"],
    settings["vehicle-classes"],
    timer,
    settings["observation-time"],
    settings["vehicle-size-coeffs"],
    len(list_region)
)

# Начало обработки видео
logging.info("Начало обработки видео...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    counter.count(frame, annotate=True)
    logging.info(f"Обработан кадр по времени {timer.time}")
    sector.update(counter.regions)
    logging.info(f"Обновлены сектора по времени {sector.period_timer.time}")

    cv2.putText(
        frame,
        f"{sector.sectors[0].classwise_traveled_count}",
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
        break

sector.new_period()
logging.info("Обработка видео завершена.")

# TODO: Максим: сделать генерацию отчета за все периоды
traffic_stats = sector.traffic_stats()
classwise_stats = sector.classwise_stats()
logging.info("Созданы датафреймы со статистикой.")
print(traffic_stats)
print(classwise_stats)

merged_stats = pd.concat([traffic_stats, classwise_stats], axis=1)
merged_stats.to_excel(report_path)
logging.info(f"Отчет сохранен в {report_path}")

# Освобождаем ресурсы
cap.release()
output.release()
cv2.destroyAllWindows()

logging.info(f"Видеофайл сохранён в {output_path}")