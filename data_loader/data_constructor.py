import cv2
import tomllib
import logging
import numpy as np
import json

from data_loader.args_loader import load_args
from data_loader.video_loader import open_video
from data_loader.data_sector import DataSector
from traffic_observer.crossroad_manager import CrossroadManager

class Settings:
    def __init__(self):
        with open("settings.toml", "rb") as f:
            toml_settings = tomllib.load(f)
            logging.info(f"Загруженные настройки: {toml_settings}")

        self.observation_time = toml_settings["observation-time"]
        self.target_width = toml_settings["target-width"]
        self.target_height = toml_settings["target-height"]
        self.vehicle_classes = toml_settings["vehicle-classes"]
        self.vehicle_size_coeffs = toml_settings["vehicle-size-coeffs"]

class DataConstructor:
    def __init__(self):
        video_path, model_path, output_path, report_path, sector_path = load_args()
        self.__video_path = video_path
        self.__model_path = model_path
        self.__output_path = output_path
        self.__report_path = report_path
        self.__sector_path = sector_path
        self.settings = Settings()

    def get_video(self) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
        cap, fps = open_video(self.__video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(self.__output_path, fourcc, fps, (self.settings.target_width, self.settings.target_height))
        return cap, output
    
    def get_crossroad_manager(self):
        temp_cap, fps = open_video(self.__video_path)
        video_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        start_lanes = self.__adapt_list_points([
            [[904,1934],[1500,1500],[1542,1544],[938,1988]],
            [[1538,1546],[1580,1598],[986,2054],[940,1994]],
            [[1580,1604],[1634,1664],[1030,2128],[986,2054]],
            [[1632,1668],[1684,1730],[1154,2146],[1032,2130]],
            [[1684,1730],[1744,1802],[1336,2144],[1154,2146]],
        ], video_width, self.settings.target_width)

        end_regions = self.__adapt_list_points([
            [[1712,342],[1496,472],[1618,604],[1856,470]],
            [[2552,798],[2810,972],[2976,886],[2700,692]],
            [[2524,1766],[2680,1610],[2920,1828],[2744,1992]],
        ], video_width, self.settings.target_width)

        return CrossroadManager(
            start_lanes,
            end_regions,
            self.settings.vehicle_classes,
            1/fps,
            self.settings.observation_time,
            self.settings.vehicle_size_coeffs,
            [self.settings.target_height, self.settings.target_width],
            self.__model_path
        )
    
    def get_output_paths(self) -> tuple[str, str]:
        return self.__report_path, self.__output_path

    def __load_sectors(self) -> list[DataSector]:
        # TODO: fix for new format
        with open(self.__sector_path, "r", encoding="utf-8") as file:
            data = json.load(file)  

        sectors = []
        for sector in data["sectors"]:
            sector_id = sector["sector_id"]
            start_points = sector["region_start"]["coords"]
            end_points = sector["region_end"]["coords"]
            lanes_points = [lane["coords"] for lane in sector["lanes"]]
            lanes_count = sector["lanes_count"]
            sector_length = sector["sector_length"]
            max_speed = sector["max_speed"]
            
            # Creating Sector object
            sector_object = DataSector(sector_id, start_points, end_points, lanes_points, lanes_count, sector_length, max_speed)
            sectors.append(sector_object)
        
        return sectors
    
    def __adapt_list_points(self, data_sectors: list[list[int]], video_width, required_width) -> list[list[int]]:
        coeff = video_width / required_width
        # This creates a new list with each point adapted
        adapted_points = [
            [self.__adapt_resolution_points(point, coeff) for point in points]
            for points in data_sectors
        ]
        
        return adapted_points


    def __adapt_resolution_points(self, points: list[int], coef) -> list[int]:
        # Преобразование к int, так как openCV не берет float
        return (np.array(points) / coef).astype(int).tolist()