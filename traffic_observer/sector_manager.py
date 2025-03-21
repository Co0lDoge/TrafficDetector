from typing import Sequence, List, Callable

import pandas as pd
import cv2
import logging

from funcs import *
from traffic_observer.period import Period
from traffic_observer.step_timer import StepTimer
from traffic_observer.regions_counter import RegionCounter
from traffic_observer.detector import Detector
from traffic_observer.regions_counter import Region

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class Sector:
    def __init__(self, vehicle_classes):
        self.periods_data: List[Period] = []
        self.ids_travel_time = {}
        self.classwise_traveled_count = {class_name: 0 for class_name in vehicle_classes}
        self.ids_start_time = {}
        self.ids_blacklist = set()
        self.travelling_ids = {}


class SectorManager:
    def __init__(
            self,
            length: int,  
            lane_count: int,
            max_speed: int,
            vehicle_classes: Sequence[str],
            time_step: int,
            observation_time: int,
            vechicle_size_coeffs: dict[str, float],
            lanes: List,
            regions: List,
            imgsize: tuple,
            model_path:str
    ):
        self.size_coeffs = vechicle_size_coeffs
        self.vehicle_classes = vehicle_classes
        self.length = length
        self.lane_count = lane_count
        self.num_sectors = len(regions)//2  
        self.observation_period = observation_time
        self.period_timer = StepTimer(time_step)
        model = YOLO(model_path)
        self.class_names=model.names

        self.detector = Detector(model, imgsize)
        #self.lane_counter = LaneCounter(lane_points=lanes, class_names=model.names)
        self.region_counter = RegionCounter(regions_points=regions, class_names=model.names)
        self.sectors = [Sector(self.vehicle_classes) for _ in range(self.num_sectors)]


    def __annotate(self, frame, annotator, box, track_id, track_class, regions: List['Region'], get_vehicle_travel_time: Callable[[int], float]):
        visited = None
        region_start: Region = regions[0]
        region_end: Region = regions[1]

        label = ""
        color=(50, 0, 0)
        label = f'ID {track_id}"'
        if track_id in region_start.counted_ids:
            color=(255, 0, 0)
            visited = "start"
        if track_id in region_end.counted_ids and visited == "start":
            color = (0, 150, 100)
            visited = "end"
        elif track_id in region_end.counted_ids:
            color = (0, 0, 255)
            visited = "end error"
        if visited is not None:
            travel_time = get_vehicle_travel_time(track_id)
            time = f"{travel_time:.2f}" if travel_time is not None else None
            label = f'ID {track_id} | {visited} | {time}"'

        annotator.box_label(box, label, color)


    def update(self, frame: cv2.typing.MatLike):
        boxes, track_ids, classes = self.detector.track(frame)

        # Обновление задержки линий
        #self.lane_counter.update(self.period_timer.step)
        #self.lane_counter.draw_line_info(frame)

        # Обработка детекций
        annotator = Annotator(frame, line_width=1, example=str(self.class_names))
        for box, track_id, track_class in zip(boxes, track_ids, classes):
            self.region_counter.count_tracklet(box, track_id, track_class)
            #self.lane_counter.count_tracklet(frame, box, track_id, track_class, draw_lanes=True)
            
            self.__annotate(frame, annotator, box, track_id, track_class, self.region_counter.regions, self.travel_time)
            self.region_counter.draw_regions(frame)
        
        logging.info(f"Обработан кадр по времени {self.period_timer.time}")

        # Обновление таймера и периода
        self.period_timer.step_forward()
        if self.period_timer.time >= self.observation_period:
            self.new_period()

        # Итерация по секторам и регионам
        self.iterate_through_regions()

        logging.info(f"Обновлены сектора по времени {self.period_timer.time}")

    def iterate_through_regions(self):
        for i in range(self.num_sectors):
            start_counter = self.region_counter.regions[2 * i]
            end_counter = self.region_counter.regions[2 * i + 1]
            sector = self.sectors[i]

            for vehicle_id in start_counter.counted_ids:
                if vehicle_id not in sector.ids_start_time and vehicle_id not in sector.ids_blacklist:
                    sector.ids_start_time[vehicle_id] = self.period_timer.unresettable_time

            for vehicle_id in end_counter.counted_ids:
                if vehicle_id not in sector.ids_blacklist and vehicle_id in sector.ids_start_time:
                    dt = self.period_timer.unresettable_time - sector.ids_start_time[vehicle_id]
                    sector.ids_start_time.pop(vehicle_id)

                    sector.ids_travel_time[vehicle_id] = dt

                    class_name = end_counter.counted_ids[vehicle_id].class_name
                    sector.classwise_traveled_count[class_name] += 1
                    sector.ids_blacklist.add(vehicle_id)

    def new_period(self):
        for sector in self.sectors:
            sector.periods_data.append(Period(
                sector.ids_travel_time.copy(),
                sector.classwise_traveled_count.copy(),
                self.period_timer.time
            ))

            sector.ids_travel_time.clear()
            sector.classwise_traveled_count = {class_name: 0 for class_name in self.vehicle_classes}
        self.period_timer.reset()

    def traffic_stats(self) -> List[pd.DataFrame]:
        dataframes = []
        for sector in self.sectors:
            stats = {
                "Интенсивность траффика": [],
                "Среднее время проезда": [],
                "Средняя скорость движения": [],
                "Плотность траффика": [],
                "Время наблюдения": []
            }
            for period in sector.periods_data:
                stats["Интенсивность траффика"].append(traffic_intensity(
                    period.classwise_traveled_count,
                    self.size_coeffs,
                    period.observation_time
                ))

                vehicles_travel_time = period.ids_travel_time.values()
                stats["Среднее время проезда"].append(mean_travel_time(vehicles_travel_time))
                stats["Средняя скорость движения"].append(mean_vehicle_speed(vehicles_travel_time, self.length))

                stats["Плотность траффика"].append(traffic_density(
                    period.classwise_traveled_count,
                    self.size_coeffs,
                    vehicles_travel_time,
                    self.length,
                    period.observation_time,
                    lane_count=self.lane_count
                ))

                stats["Время наблюдения"].append(period.observation_time)
            dataframes.append(pd.DataFrame(stats))

        return dataframes

    def classwise_stats(self) -> List[pd.DataFrame]:
        dataframes = []
        for sector in self.sectors:
            stats = {class_name: [] for class_name in self.vehicle_classes}
            for period in sector.periods_data:
                for class_name in self.vehicle_classes:
                    stats[class_name].append(period.classwise_traveled_count[class_name])
            dataframes.append(pd.DataFrame(stats))

        return dataframes
    
    def travel_time(self, vehicle_id: int) -> float:
        for sector in self.sectors:
            if vehicle_id in sector.ids_start_time:
                return self.period_timer.unresettable_time - sector.ids_start_time[vehicle_id]
            elif vehicle_id in sector.ids_travel_time:
                return sector.ids_travel_time[vehicle_id]
        return None
