import cv2
import tomllib
import logging
import numpy as np
import json

from data_loader.args_loader import load_args
from data_loader.video_loader import open_video

class DataSector:
    def __init__(self, sector_id, start_points, end_points, lanes_points, lanes_count, sector_length, max_speed):
        self.id = sector_id
        self.start_points = start_points
        self.end_points = end_points
        self.lanes_points = lanes_points
        self.lanes_count = lanes_count
        self.sector_length = sector_length
        self.max_speed = max_speed

class DataConstructor:
    def __init__(self):
        video_path, model_path, output_path, report_path, sector_path = load_args()
        self.__video_path = video_path
        self.__model_path = model_path
        self.__output_path = output_path
        self.__report_path = report_path
        self.__sector_path = sector_path
        with open("settings.toml", "rb") as f:
            self.settings = tomllib.load(f)
            logging.info(f"Загруженные настройки: {self.settings}")
        
        cap, fps = open_video(self.__video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(self.__output_path, fourcc, fps, (self.settings["target-width"], self.settings["target-height"]))
        
        # parse sector
        data_sectors = self.__load_sectors()
        data_sectors = self.__adapt_sectors_points(data_sectors, video_width, self.settings["target-width"])

    def __load_sectors(self) -> list[DataSector]:
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
    
    def __adapt_sectors_points(self, data_sectors, video_width, required_width) -> list[list[int]]:
        # Адаптирует список точек региона к необходимому разрешению
        coeff = video_width / required_width

        for sector in data_sectors:
            sector.start_points = self.__adapt_resolution_points(sector.start_points, coeff)
            sector.end_points = self.__adapt_resolution_points(sector.end_points, coeff)
            for lane in sector.lanes_points:
                lane = self.__adapt_resolution_points(lane, coeff)

    def __adapt_resolution_points(self, points: list[int], coef) -> list[int]:
        # Преобразование к int, так как openCV не берет float
        return (np.array(points) / coef).astype(int).tolist()