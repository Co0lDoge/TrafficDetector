## Установка зависимостей
Рекомендуемая версия python: 3.11.9
```sh
pip install -r requirements.txt
```

## Настройка
Настройки находятся в файле `settings.toml`.
```sh
# Время наблюдения за сектором. В секундах
observation-time = 30    
# Желаемое разрешение видео
target-width = 1280
target-height = 720
vehicle-classes = ["bus", "car", "motobike", "road_train", "truck"]
# Коэффиценты привидения
vehicle-size-coeffs = { "car" = 1, "motorbike" = 0.5, "truck" = 1.8, "road_train" = 2.7, "bus" = 2.2 }
```

## Запуск в виде модуля
```sh
from directional_detector import directional_detect

directional_detect(
    video_path = "__test/video/galle_cross_quickstart.mp4",
    model_path = "__test/model/detector_yolov10s.pt",
    output_path = "__test/output/output_video.mp4",
    report_path = "__test/output/report.json",
    sector_path = "__test/region/multiregion_galle.json",
)
```

## Запуск в виде приложения
```sh
python directional_detector.py 
--video-path video/test_720p.mp4 
--model-path model/yolov8s_1280_720.pt 
--output-path output/order-479.mp4 
--report-path output/traffic-stats.xlsx 
--sector_path regions.json
```

## При использовании модели OpenVINO путь необходимо указывать к директории со всеми файлами модели
```sh
--model-path model/yolov10s_openvino_model/
```
