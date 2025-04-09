from typing import Sequence, List, Callable

import pandas as pd
import cv2
import logging

from funcs import *
from traffic_observer.period import Period
from traffic_observer.step_timer import StepTimer
from traffic_observer.region import Region
from traffic_observer.detector import Detector
from traffic_observer.lane import Lane

from data_loader.data_sector import DataSector
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class Direction:
    def __init__(self, data_sector: DataSector, vehicle_classes):
        self.start_region: Region = Region(data_sector.start_points)
        self.lanes: list[Lane] = [Lane(lane_points) for lane_points in data_sector.lanes_points]
        self.lanes_count: int = data_sector.lanes_count
        self.length: int = data_sector.sector_length
        self.max_speed: int = data_sector.max_speed
        self.periods_data: List[Period] = []
        self.ids_travel_time = {}
        self.ids_free_time = {}
        self.classwise_traveled_count = {class_name: 0 for class_name in vehicle_classes}
        self.ids_start_time = {}
        self.ids_blacklist = set()

class CrossroadManager:
    def __init__(
            self,
            start_lanes: list[list[int]],
            end_regions: list[list[int]],
            vehicle_classes: Sequence[str],
            time_step: int,
            observation_time: int,
            vechicle_size_coeffs: dict[str, float],
            imgsize: tuple,
            model_path:str
    ):
        self.size_coeffs = vechicle_size_coeffs
        self.vehicle_classes = vehicle_classes
        self.observation_period = observation_time
        self.period_timer = StepTimer(time_step)
        model = YOLO(model_path)
        self.class_names=model.names

        self.detector = Detector(model, imgsize)
        self.start_lanes = [Region(points) for points in start_lanes]
        self.end_regions = [Region(points) for points in end_regions]

    def __annotate(self, im0, annotator, box, track_id, cls):
        annotator.box_label(box, "", color=(255, 0, 0))

    def __annotate_debug(self, frame, annotator, box, track_id, track_class):
        color=(50, 0, 0)
        label = f'ID {track_id}"'

        for lane in self.start_lanes:
            if track_id in lane.counted_ids.keys():
                color=(255, 0, 0)
                label = f'ID {track_id} | Lane: {self.start_lanes.index(lane)}"'

        annotator.box_label(box, label, color)


    def update(self, frame: cv2.typing.MatLike):
        boxes, track_ids, classes = self.detector.track(frame)
        
        # Поиск пересечений с началом пути
        annotator = Annotator(frame, line_width=1, example=str(self.class_names))
        for box, track_id, track_class in zip(boxes, track_ids, classes):
            self.__annotate_debug(frame, annotator, box, track_id, track_class)
            for lane in self.start_lanes:
                lane.count_tracklet(box, track_id, track_class)
        
        # Поиск пересечений отслеживаемого транспорта с концами пути
        for lane in self.start_lanes:
            for vehicle_id in lane.counted_ids.keys():
                pass

            

        # Поиск пересечений с концом пути
        # for track_id in region in all self.end_regions:
        
        for lane in self.start_lanes:
            lane.draw_regions(frame)
        for lane in self.end_regions:
            lane.draw_regions(frame) # TODO: Change color to red

    def __get_vehicle_travel_time_debug(self, vehicle_id: int) -> float:
        # Get travel time for a vehicle by its ID
        return None
