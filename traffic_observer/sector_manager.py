from typing import Sequence, List

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

class Sector:
    def __init__(self, data_sector: DataSector, vehicle_classes):
        self.start_region: Region = Region(data_sector.start_points)
        self.end_region: Region = Region(data_sector.end_points)
        self.lanes: list[Lane] = [Lane(lane_points) for lane_points in data_sector.lanes_points]
        self.lanes_count: int = data_sector.lanes_count
        self.length: int = data_sector.sector_length
        self.max_speed: int = data_sector.max_speed
        self.periods_data: List[Period] = []
        self.ids_travel_time = {}
        self.classwise_traveled_count = {class_name: 0 for class_name in vehicle_classes}
        self.ids_start_time = {}
        self.ids_blacklist = set()
        self.travelling_ids = {}

class SectorManager:
    def __init__(
            self,
            data_sectors: list[DataSector],
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
        self.sectors = [Sector(data_sector, self.vehicle_classes) for data_sector in data_sectors]

    def __annotate(self, im0, annotator, box, track_id, cls):
        annotator.box_label(box, "", color=(255, 0, 0))


    def update(self, frame: cv2.typing.MatLike):
        boxes, track_ids, classes = self.detector.track(frame)

        # Обработка детекций
        annotator = Annotator(frame, line_width=1, example=str(self.class_names))
        for box, track_id, track_class in zip(boxes, track_ids, classes):
            for sector in self.sectors:
                # TODO: make method for those
                sector.start_region.count_tracklet(box, track_id, track_class)
                sector.end_region.count_tracklet(box, track_id, track_class)
                sector.start_region.draw_regions(frame)
                sector.end_region.draw_regions(frame)
                for lane in sector.lanes:
                    lane.draw_lane(frame)
            
            self.__annotate(frame, annotator, box, track_id, track_class)
 
        logging.info(f"Обработан кадр по времени {self.period_timer.time}")

        # Обновление таймера и периода
        self.period_timer.step_forward()
        if self.period_timer.time >= self.observation_period:
            self.new_period()

        # Итерация по секторам и регионам
        self.iterate_through_regions()

        # Обработка линий
        for sector in self.sectors:
            for lane in sector.lanes:
                lane.delay += self.period_timer.step
                for vehicle_id in sector.travelling_ids:
                    lane.count_tracklet(sector.travelling_ids[vehicle_id], vehicle_id)

        logging.info(f"Обновлены сектора по времени {self.period_timer.time}")

    def iterate_through_regions(self):
        for sector in self.sectors:
            start_counter = sector.start_region
            end_counter = sector.end_region

            for vehicle_id in start_counter.counted_ids:
                if vehicle_id not in sector.ids_start_time and vehicle_id not in sector.ids_blacklist:
                    sector.ids_start_time[vehicle_id] = self.period_timer.unresettable_time

            for vehicle_id in end_counter.counted_ids:
                if vehicle_id not in sector.ids_blacklist and vehicle_id in sector.ids_start_time:
                    dt = self.period_timer.unresettable_time - sector.ids_start_time[vehicle_id]
                    sector.ids_start_time.pop(vehicle_id)

                    sector.ids_travel_time[vehicle_id] = dt

                    track_class = end_counter.counted_ids[vehicle_id].track_class
                    class_name = self.class_names[track_class]
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

        for region in self.sectors: # TODO: use another more frequently called method for long periods of time
            region.start_region.counted_ids.clear()
            region.end_region.counted_ids.clear()

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
                stats["Средняя скорость движения"].append(mean_vehicle_speed(vehicles_travel_time, sector.length))

                stats["Плотность траффика"].append(traffic_density(
                    period.classwise_traveled_count,
                    self.size_coeffs,
                    vehicles_travel_time,
                    sector.length,
                    period.observation_time,
                    lane_count=sector.lanes_count
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
