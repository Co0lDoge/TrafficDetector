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

    if(len(list_region) % 2 != 0):
        raise ValueError("Количество регионов не чётное")

    # Доступ к аргументам
    video_path = args.video_path
    model_path = args.model_path
    output_path = args.output_path
    report_path = args.report_path
    list_region = list_region

    return video_path, model_path, output_path, report_path, list_region

def get_adapted_region_points(regions, video_width, required_width) -> list[list[int]]:
    coeff = video_width / required_width
    region_points = []
    
    for idx, region in enumerate(regions):
        coordinates = region['coordinates']
        
        # Преобразование к int, так как openCV не берет float
        points = resolution_adapt(coordinates, coeff)
        region_points.append(points)
        
    return region_points

def resolution_adapt(points: list[int], coef) -> list[int]:
    # Преобразование к int, так как openCV не берет float
    return (np.array([[coord['x'], coord['y']] for coord in points]) / coef).astype(int).tolist()