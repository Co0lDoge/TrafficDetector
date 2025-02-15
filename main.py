import cv2
import tomllib
import logging

from args_loader import load_args, get_adapted_region_points
from data_manager.traffic_report import create_stats_report
from traffic_observer.sector import SectorManager
from traffic_observer.regions_counter import RegionsCounter

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

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

video_path, model_path, output_path, report_path, list_region = load_args()

with open("settings.toml", "rb") as f:
    settings = tomllib.load(f)
logging.info(f"Загруженные настройки: {settings}")

cap, fps = open_video(video_path)

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
width, height = settings["target-width"], settings["target-height"]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

regions = get_adapted_region_points(list_region, video_width, width)

counter = RegionsCounter(model_path, regions_points=regions, imgsz=(height, width))
sector = SectorManager(
    settings["sector-length"],
    settings["lane-count"],
    settings["vehicle-classes"],
    1/fps,
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
    #logging.info(f"Обработан кадр по времени {timer.time}")
    sector.update(counter.regions)
    logging.info(f"Обновлены сектора по времени {sector.period_timer.time}")

    # Показ текущего кадра
    cv2.imshow("Frame", frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sector.new_period()
logging.info("Обработка видео завершена.")

# Создание отчёта
create_stats_report(sector, report_path)

# Освобождаем ресурсы
cap.release()
output.release()
cv2.destroyAllWindows()

logging.info(f"Видеофайл сохранён в {output_path}")