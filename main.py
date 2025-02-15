import cv2
import tomllib
import pandas as pd
from pandas.io.excel import ExcelWriter
import logging

from args_loader import load_args, get_adapted_region_points
from sector import SectorCluster
from regions_counter import RegionsCounter
from step_timer import StepTimer

def create_stats_report(sector_cluster: SectorCluster, report_path: str):
    traffic_stats = sector_cluster.traffic_stats()
    classwise_stats = sector_cluster.classwise_stats()
    logging.info("Созданы датафреймы со статистикой.")

    res_dataframes = []
    i = 1
    for traf_stat, class_stat in zip(traffic_stats, classwise_stats):
        df_res_tmp = pd.concat([traf_stat, class_stat], axis=1)
        res_dataframes.append(df_res_tmp)

        print("*********************")
        print(f"Sector #{i}")
        print(df_res_tmp)
        i += 1

    # Запись данных в файл
    with ExcelWriter(report_path) as writer:
        for ind, df in enumerate(res_dataframes):
            df.to_excel(writer, sheet_name=f"{ind + 1}")

    # Освобождаем ресурсы
    cap.release()
    output.release()
    cv2.destroyAllWindows()

    logging.info(f"Видеофайл сохранён в {output_path}")

def get_fps(cap) -> float|int:
    major_ver, _, _ = cv2.__version__.split('.')
    if int(major_ver) >= 3:
        return cap.get(cv2.CAP_PROP_FPS)
    return cap.get(cv2.cv.CV_CAP_PROP_FPS)

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
counter = RegionsCounter(model_path, regions_points=regions, imgsz=(height, width))

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

    # Показ текущего кадра
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sector.new_period()
logging.info("Обработка видео завершена.")

# Создание отчёта
create_stats_report(sector, report_path)