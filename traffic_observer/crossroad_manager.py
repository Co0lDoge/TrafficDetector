from typing import Sequence
import cv2
import logging

from data_manager.traffic_funcs import *
from traffic_observer.component.step_timer import StepTimer
from traffic_observer.component.region import Region
from traffic_observer.component.detector import Detector
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class TrackedVehicle:
    def __init__(self, class_name: str, bbox, direction_id: int, lane_id: int):
        self.track_class = class_name
        self.bbox = bbox
        self.direction_id = direction_id
        self.lane_id = lane_id
        self.start_delay = 0.0
        self.travel_time = 0.0
        self.end_id = None 

class TrackHolder():
    ''' Holds tracked vehicles moving from start directions '''
    def __init__(self):
        self.tracked_vehicles: dict[int, TrackedVehicle] = {}

class Direction:
    ''' Holds points of the lines of direction '''
    def __init__(self, start_lanes):
        self.lanes = [Region(points) for points in start_lanes]

class DataCollector:
    ''' Saves the data of vehicles that passed the end region '''
    def __init__(self):
        self.collected_vehicles: list[TrackedVehicle] = []

    def add_vehicle(self, vehicle: TrackedVehicle):
        self.collected_vehicles.append(vehicle)

class CrossroadManager:
    ''' Manages the tracking of vehicles by directions 
    from which they enter and where they exit '''
    def __init__(
            self,
            directions_lanes: list[list[int]],
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
        self.tracker_manager = TrackHolder()
        self.datacollector = DataCollector()
        self.start_directions = [Direction(lanes) for lanes in directions_lanes]
        self.end_regions = [Region(points) for points in end_regions]

    def __annotate_debug(self, annotator, box, track_id, track_class):
        color=(50, 0, 0)
        #class_name = self.vehicle_classes[int(track_class)]
        
        label = f'ID {track_id}"'
        if track_id in self.tracker_manager.tracked_vehicles:
            color=(255, 0, 0)
            lane_id = self.tracker_manager.tracked_vehicles[track_id].lane_id
            start_delay = f"{self.tracker_manager.tracked_vehicles[track_id].start_delay:.2f}"
            travel_time = f"{self.tracker_manager.tracked_vehicles[track_id].travel_time:.2f}"
            label = f'ID {track_id} | Lane: {lane_id} | {start_delay} | {travel_time}"'

        annotator.box_label(box, label, color)

    def update(self, frame: cv2.typing.MatLike):
        boxes, track_ids, classes = self.detector.track(frame)
        
        # Поиск пересечений с началом пути
        annotator = Annotator(frame, line_width=1, example=str(self.class_names))
        for box, track_id, track_class in zip(boxes, track_ids, classes):
            self.__annotate_debug(annotator, box, track_id, track_class)

            # Check tracked ids
            if track_id in self.tracker_manager.tracked_vehicles:
                tracked_vehicle = self.tracker_manager.tracked_vehicles[track_id]
                direction = self.start_directions[tracked_vehicle.direction_id] 
                if direction.lanes[tracked_vehicle.lane_id].is_intersected(box):
                    tracked_vehicle.start_delay += self.period_timer.step
                else:
                    tracked_vehicle.travel_time += self.period_timer.step

                for region in self.end_regions:
                    if region.is_intersected(box):
                        tracked_vehicle.end_id = self.end_regions.index(region)
                        self.datacollector.add_vehicle(tracked_vehicle)
                        self.tracker_manager.tracked_vehicles.pop(track_id)
                continue

            # Check untracked ids
            for direction in self.start_directions:
                for lane in direction.lanes:
                    if lane.is_intersected(box):
                        direction_id = self.start_directions.index(direction)
                        lane_id = direction.lanes.index(lane)
                        self.tracker_manager.tracked_vehicles[track_id] = TrackedVehicle(track_class, box, direction_id, lane_id)
        
        # Drawing region borders
        for direction in self.start_directions:
            for lane in direction.lanes:
                lane.draw_regions(frame)

        for lane in self.end_regions:
            lane.draw_regions(frame, text = str(self.end_regions.index(lane)), color = (0, 0, 255))

        logging.info("Frame processed")

    def __get_vehicle_travel_time_debug(self, vehicle_id: int) -> float:
        # Get travel time for a vehicle by its ID
        return None
