import cv2
import tomllib
import logging
import numpy as np

from data_loader.component.args_loader import load_args
from data_loader.component.video_loader import open_video
from data_loader.component.region_loader import load_json_region
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
    def __init__(self, video_path, model_path, output_path, report_path, sector_path):
        self.__video_path = video_path
        self.__model_path = model_path
        self.__output_path = output_path
        self.__report_path = report_path
        self.__sector_path = sector_path
        self.settings = Settings()

    def from_args():
        video_path, model_path, output_path, report_path, sector_path = load_args()
        return DataConstructor(
            video_path,
            model_path,
            output_path,
            report_path,
            sector_path
        )

    def get_video(self) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
        cap, fps = open_video(self.__video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(self.__output_path, fourcc, fps, (self.settings.target_width, self.settings.target_height))
        return cap, output
    
    def get_crossroad_manager(self):
        temp_cap, fps = open_video(self.__video_path)
        video_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        directions, end_regions = self.get_adapted_regions(video_width, self.settings.target_width)

        return CrossroadManager(
            directions,
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
    
    def get_adapted_regions(self, video_width, target_width):
        regions_data = load_json_region(self.__sector_path)

        # Process directions.
        directions_processed = []
        for direction in regions_data.get('directions', []):
            adapted_direction = self.__adapt_list_points(direction, video_width, target_width)
            directions_processed.append(adapted_direction)
        
        # Process end_region.
        end_region_data = regions_data.get('end_region')
        if end_region_data:
            adapted_end_region = self.__adapt_list_points(end_region_data, video_width, target_width)
        else:
            adapted_end_region = None
            logging.info("No end_region found in JSON.")

        return [
            directions_processed,
            adapted_end_region
        ]

    
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