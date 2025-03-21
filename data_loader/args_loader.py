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
    parser.add_argument("--sector_data", type=str, required=True, help="Массив точек областей")

    # Получение всех аргументов
    args = parser.parse_args()

    # Парсинг регионов
    with open(args.sector_data, "r", encoding="utf-8") as file:
        data = json.load(file)  # вместо json.loads()

    # Разбираем данные
    for sector in data["sectors"]:
        sector_id = sector["sector_id"]
        region_start = sector["region_start"]["coords"]
        region_end = sector["region_end"]["coords"]
        lanes = [lane["coords"] for lane in sector["lanes"]]
        lanes_count = sector["lanes_count"]
        
        print(f"Sector ID: {sector_id}")
        print(f"Region Start: {region_start}")
        print(f"Region End: {region_end}")
        print(f"Lanes: {lanes}")
        print(f"Lanes Count: {lanes_count}")
        print("-" * 40)

    # Доступ к аргументам
    video_path = args.video_path
    model_path = args.model_path
    output_path = args.output_path
    report_path = args.report_path
    sector_data = sector_data
    
    return video_path, model_path, output_path, report_path, sector_data

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