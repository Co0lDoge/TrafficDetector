import cv2
import logging

from data_manager.traffic_report import create_stats_report
from data_loader.data_constructor import DataConstructor

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

dataConstructor = DataConstructor()
cap, output = dataConstructor.get_video()
sector_manager = dataConstructor.get_sector_manager()
settings = dataConstructor.settings

# Начало обработки видео
logging.info("Начало обработки видео...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (settings.target_width, settings.target_height))
    sector_manager.update(frame)

    # Показ текущего кадра
    cv2.imshow("frame", frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

report_path, output_path  = dataConstructor.get_output_paths()

# Освобождаем ресурсы
sector_manager.new_period()
logging.info("Обработка видео завершена.")

# Сохранение видеофайла
cap.release()
output.release()
cv2.destroyAllWindows()

logging.info(f"Видеофайл сохранён в {output_path}")

# Создание отчёта
create_stats_report(sector_manager, report_path)
