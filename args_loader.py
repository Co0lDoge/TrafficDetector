import argparse
import json
import numpy as np

def load_args():
    # Добавление аргументов запуска
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True, help="Путь к видео")
    parser.add_argument("--model-path", type=str, required=True, help="Путь к модельке")
    parser.add_argument("--output-path", type=str, required=True, help="Путь для выходного файлы")
    parser.add_argument("--report-path", type=str, required=True, help="Путь для выходного отчета")
    parser.add_argument("--regions", type=str, required=True, help="Массив точек областей")

    # Получение всех аргументов
    args = parser.parse_args()

    # Парсинг регионов
    if args.regions:
        with open(args.regions, 'r') as f:
            list_region = json.load(f)
    else:
        list_region = json.loads(args.list_region)
    print(f"Regions: {list_region}")

    # Доступ к аргументам
    video_path = args.video_path
    model_path = args.model_path
    output_path = args.output_path
    report_path = args.report_path
    list_region = list_region

    return video_path, model_path, output_path, report_path, list_region

def region_adapt(regions, video_width, required_width):
    coeff = video_width / required_width
    adapted = {}
    
    for idx, region in enumerate(regions):
        area_number = region['areaNumber']  # TODO: возможно нужно будет использовать
        coordinates = region['coordinates']
        
        # Преобразование к int, так как openCV не берет float
        points = (np.array([[coord['x'], coord['y']] for coord in coordinates]) / coeff).astype(int).tolist()
        
        # Первый регион - начало, последний - конец, остальные - middle_X
        if idx == 0:
            adapted["start"] = points
        elif idx == len(regions) - 1:
            adapted["end"] = points
        else:
            adapted[f"middle_{idx}"] = points

    return adapted