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

class TrackedVehicle:
    def __init__(self, class_name: str, bbox):
        self.track_class = class_name
        self.bbox = bbox

class Direction:
    def __init__(self, start_lanes, end_regions):
        self.start_lanes = [Region(points) for points in start_lanes]
        self.end_regions = [Region(points) for points in end_regions]
        self.tracked_vehicled: dict[int, TrackedVehicle] = {} # track_id -> vehicle_id

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
        self.direction = Direction(start_lanes, end_regions)

    def __annotate_debug(self, annotator, box, track_id, track_class):
        color=(50, 0, 0)
        label = f'ID {track_id}"'

        if track_id in self.direction.tracked_vehicled:
            color=(255, 0, 0)
            label = f'ID {track_id} | Lane: {...}"'

        annotator.box_label(box, label, color)

    def update(self, frame: cv2.typing.MatLike):
        boxes, track_ids, classes = self.detector.track(frame)
        
        # Поиск пересечений с началом пути
        annotator = Annotator(frame, line_width=1, example=str(self.class_names))
        for box, track_id, track_class in zip(boxes, track_ids, classes):
            self.__annotate_debug(annotator, box, track_id, track_class)

            # Check tracked ids
            if track_id in self.direction.tracked_vehicled:
                for region in self.direction.end_regions:
                    if region.is_intersected(box):
                        self.direction.tracked_vehicled.pop(track_id)

            # Check untracked ids
            for lane in self.direction.start_lanes:
                if lane.is_intersected(box):
                    self.direction.tracked_vehicled[track_id] = TrackedVehicle(track_class, box)
        
        # Drawing region borders
        for lane in self.direction.start_lanes:
            lane.draw_regions(frame)
        for lane in self.direction.end_regions:
            lane.draw_regions(frame) # TODO: Change color to red

    def __get_vehicle_travel_time_debug(self, vehicle_id: int) -> float:
        # Get travel time for a vehicle by its ID
        return None
