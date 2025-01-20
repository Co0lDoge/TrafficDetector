## Установка зависимостей
Рекомендуемая версия python: 3.11.9
```sh
pip install -r requirements.txt
```

## Настройка
Настройки находятся в файле `settings.toml`.
```sh
sector-length = 0.1    # По моим расчетам. В километрах
observation-time = 30    # Время наблюдения за сектором. В секундах
lane-count = 1    # Кол-во полос
# Желаемое разрешение видео
target-width = 1280
target-height = 720
vehicle-classes = ["bus", "car", "motobike", "road_train", "truck"]
# Коэффиценты привидения
vehicle-size-coeffs = { "car" = 1, "motorbike" = 0.5, "truck" = 1.8, "road_train" = 2.7, "bus" = 2.2 }
```

## Запуск
```sh
python main.py 
--video-path video/test_720p.mp4 
--model-path model/yolov8s_1280_720.pt 
--output-path output/order-479.mp4 
--report-path output/traffic-stats.xlsx 
--regions regions.json
```
## При использовании модели OpenVINO путь необходимо указывать к директории со всеми файлами модели
```sh
--model-path model/yolov10s_openvino_model/
```
