import cv2
import tomllib
import logging
import numpy as np
import json

from data_loader.args_loader import load_args
from data_loader.video_loader import open_video

class DataSector:
    def __init__(self, sector_id, start_points, end_points, lanes_points, lanes_count):
        self.id = sector_id
        self.start_points = start_points
        self.end_points = end_points
        self.lanes_points = lanes_points
        self.lanes_count = lanes_count

class DataConstructor:
    def __init__(self):
        video_path, model_path, output_path, report_path, sector_path = load_args()

        with open("settings.toml", "rb") as f:
            settings = tomllib.load(f)
        logging.info(f"Загруженные настройки: {settings}")

        cap, fps = open_video(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        width, height = settings["target-width"], settings["target-height"]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # parse sector
        data_sectors = self.__load_sectors(sector_path)

        # adapt sector's region points
        data_sectors = self.__get_adapted_regions_of_sector(data_sectors, video_width, width)
        
        # sector = self.__get_adapted_region_points(list_region, video_width, width)

    def __load_sectors(self, sector_data) -> list[DataSector]:
        with open(sector_data, "r", encoding="utf-8") as file:
            data = json.load(file)  

        sectors = []
        for sector in data["sectors"]:
            sector_id = sector["sector_id"]
            start_points = sector["region_start"]["coords"]
            end_points = sector["region_end"]["coords"]
            lanes_points = [lane["coords"] for lane in sector["lanes"]]
            lanes_count = sector["lanes_count"]
            
            # Creating Sector object
            sector_object = DataSector(sector_id, start_points, end_points, lanes_points, lanes_count)
            
            # Appending the sector object to the sectors list
            sectors.append(sector_object)
        return sectors
    
    def __get_adapted_regions_of_sector(self, data_sectors, video_width, required_width) -> list[list[int]]:
        # Адаптирует список точек региона к необходимому разрешению
        coeff = video_width / required_width

        for sector in data_sectors:
            sector.start_points = self._adapt_resolution_points(sector.start_points, coeff)
            sector.end_points = self._adapt_resolution_points(sector.end_points, coeff)
            for lane in sector.lanes_points:
                lane = self._adapt_resolution_points(lane, coeff)

    def _adapt_resolution_points(self, points: list[int], coef) -> list[int]:
        # Преобразование к int, так как openCV не берет float
        return (np.array(points) / coef).astype(int).tolist()