import cv2
import logging

from data_manager.traffic_report import create_stats_report
from data_loader.data_constructor import DataConstructor
from traffic_observer.sector_manager import SectorManager

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

dataConstructor = DataConstructor()
cap, output = dataConstructor.get_video()
# sector_manager = SectorManager(
#     settings["sector-length"],
#     settings["lane-count"], # TODO: lane rework for each sector
#     settings["max-speed"],
#     settings["vehicle-classes"],
#     1/fps,
#     settings["observation-time"],
#     settings["vehicle-size-coeffs"],
#     lanes = [], # TODO: replace with non-placeholder
#     regions = regions,
#     imgsize = [height, width],
#     model_path = model_path
# )

# # Начало обработки видео
# logging.info("Начало обработки видео...")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (width, height))
#     sector_manager.update(frame)

#     # Показ текущего кадра
#     cv2.imshow("frame", frame)
#     output.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# sector_manager.new_period()
# logging.info("Обработка видео завершена.")

# # Создание отчёта
# create_stats_report(sector_manager, report_path)

# # Освобождаем ресурсы
# cap.release()
# output.release()
# cv2.destroyAllWindows()

# logging.info(f"Видеофайл сохранён в {output_path}")