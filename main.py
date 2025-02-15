import cv2
import tomllib
import logging

from data_loader.args_loader import load_args, get_adapted_region_points
from data_loader.video_loader import open_video
from data_manager.traffic_report import create_stats_report
from traffic_observer.sector import SectorManager
from traffic_observer.regions_counter import RegionCounter

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

counter = RegionCounter(model_path, regions_points=regions, imgsz=(height, width))
sector_manager = SectorManager(
    settings["sector-length"],
    settings["lane-count"],
    settings["vehicle-classes"],
    1/fps,
    settings["observation-time"],
    settings["vehicle-size-coeffs"],
    counter
)

# Начало обработки видео
logging.info("Начало обработки видео...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    sector_manager.update(frame)

    # Показ текущего кадра
    cv2.imshow("Frame", frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sector_manager.new_period()
logging.info("Обработка видео завершена.")

# Создание отчёта
create_stats_report(sector_manager, report_path)

# Освобождаем ресурсы
cap.release()
output.release()
cv2.destroyAllWindows()

logging.info(f"Видеофайл сохранён в {output_path}")