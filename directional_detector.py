import cv2
import logging

import data_manager.report_directional as report_manager
from data_loader.data_constructor import DataConstructor

def __run_detector(
    load_video_path = None,
    load_model_path = None,
    load_output_path = None,
    load_report_path = None,
    load_sector_path = None,
    use_args = False,
    show_image = False
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    if use_args:
        dataConstructor = DataConstructor.from_args()
    else:
        dataConstructor = DataConstructor(
            load_video_path, 
            load_model_path,
            load_output_path,
            load_report_path,
            load_sector_path
        )
    
    cap, output = dataConstructor.get_video()
    traffic_manager = dataConstructor.get_crossroad_manager()
    settings = dataConstructor.settings
    # TODO: add check for report format before script execution

    # Начало обработки видео
    logging.info("Начало обработки видео...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (settings.target_width, settings.target_height))
        traffic_manager.update(frame)

        # Показ текущего кадра
        if show_image:
            cv2.imshow("frame", frame)

        output.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    report_path, output_path  = dataConstructor.get_output_paths()

    # Освобождаем ресурсы
    logging.info("Обработка видео завершена.")

    # Сохранение видеофайла
    cap.release()
    output.release()
    cv2.destroyAllWindows()

    logging.info(f"Видеофайл сохранён в {output_path}")

    # Создание отчёта
    report_manager.create_report(traffic_manager.datacollector, report_path)

def directional_detect(
    video_path,
    model_path,
    output_path,
    report_path,
    sector_path,
    show_image = False
):
    ''' Produces annotated video and traffic report '''
    __run_detector(
        video_path,
        model_path,
        output_path,
        report_path,
        sector_path,
        show_image=show_image
    )

if __name__ == "__main__":
    __run_detector(use_args=True, show_image=True)
