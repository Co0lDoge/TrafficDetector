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

class Sector:
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

    def __annotate_debug(self, frame, annotator, box, track_id, track_class, sector: Sector, get_vehicle_travel_time: Callable[[int], float]):
        visited = None

        label = ""
        color=(50, 0, 0)
        label = f'ID {track_id}"'
        if track_id in sector.ids_start_time:
            color=(255, 0, 0)
            visited = "start"
        if track_id in sector.ids_travel_time:
            color = (0, 150, 100)
            visited = "end"
        if visited is not None:
            travel_time = get_vehicle_travel_time(track_id)
            time = f"{travel_time:.2f}" if travel_time is not None else None
            label = f'ID {track_id} | {visited} | {time}"'

        annotator.box_label(box, label, color)


    def update(self, frame: cv2.typing.MatLike):
        boxes, track_ids, classes = self.detector.track(frame)

        # Обработка детекций
        annotator = Annotator(frame, line_width=1, example=str(self.class_names))
        for box, track_id, track_class in zip(boxes, track_ids, classes):
            for sector in self.sectors:
                # TODO: make method for those
                # TODO optimize: count tracket only for start regions
                # if tracklet is not tracked in sector, then only in end region
                sector.start_region.count_tracklet(box, track_id, track_class)
                sector.start_region.draw_regions(frame)
                for lane in sector.lanes:
                    lane.draw_lane(frame)
            
            #self.__annotate(frame, annotator, box, track_id, track_class)
            self.__annotate_debug(frame, annotator, box, track_id, track_class, sector, self.__get_vehicle_travel_time_debug)
 
        logging.info(f"Обработан кадр по времени {self.period_timer.time}")

        # Обновление таймера и периода
        self.period_timer.step_forward()
        if self.period_timer.time >= self.observation_period:
            self.new_period()

        # Итерация по секторам и регионам
        self.__iterate_through_regions()

        # Обработка линий
        self.__update_lanes(boxes, track_ids)

        # Итерация по линиям
        self.__iterate_through_lanes(classes, track_ids)

        logging.info(f"Обновлены сектора по времени {self.period_timer.time}")

    def __update_lanes(self, boxes, track_ids):
        # Update delay and tracklet intersections for each line in each sector
        # Must be called after __iterate_through_regions as it relies on the data formed in it
        for sector in self.sectors:
            for lane in sector.lanes:
                lane.delay += self.period_timer.step
                for vehicle_id in sector.ids_start_time.keys():
                    if vehicle_id in track_ids:
                        lane.count_tracklet(boxes[track_ids.index(vehicle_id)], vehicle_id)

    def __iterate_through_regions(self):
        # Iterate through all sectors and regions to update travel times and vehicle tracking status
        for sector in self.sectors:
            start_counter = sector.start_region

            for vehicle_id in start_counter.counted_ids:
                if vehicle_id not in sector.ids_start_time and vehicle_id not in sector.ids_blacklist:
                    sector.ids_start_time[vehicle_id] = self.period_timer.unresettable_time
        
    def __iterate_through_lanes(self, classes, track_ids):
        for sector in self.sectors:
            for lane in sector.lanes:
                for vehicle_id in lane.counted_ids:
                    if vehicle_id not in sector.ids_blacklist and vehicle_id in sector.ids_start_time:
                        dt = self.period_timer.unresettable_time - sector.ids_start_time[vehicle_id]
                        sector.ids_start_time.pop(vehicle_id)

                        sector.ids_travel_time[vehicle_id] = dt

                        # Update free travel time
                        if lane.delay < 10:
                            sector.ids_free_time[vehicle_id] = dt
                        lane.delay = 0

                        track_class = classes[track_ids.index(vehicle_id)]
                        class_name = self.class_names[track_class]
                        sector.classwise_traveled_count[class_name] += 1
                        sector.ids_blacklist.add(vehicle_id)         

    def new_period(self):
        # Reset the period timer and store the data for each sector
        for sector in self.sectors:
            sector.periods_data.append(Period(
                sector.ids_travel_time.copy(),
                sector.classwise_traveled_count.copy(),
                sector.ids_free_time.copy(),
                self.period_timer.time
            ))

            sector.ids_travel_time.clear()
            sector.ids_free_time.copy()
            sector.classwise_traveled_count = {class_name: 0 for class_name in self.vehicle_classes}
        self.period_timer.reset()

        for sector in self.sectors: # TODO: use another more frequently called method for long periods of time
            sector.start_region.counted_ids.clear()
            for lane in sector.lanes:
                lane.counted_ids.clear()
            
    def traffic_stats(self) -> List[pd.DataFrame]:
        dataframes = []
        for sector in self.sectors:
            stats = {
                "Интенсивность траффика": [],
                "Среднее время проезда сек": [],
                "Средняя скорость движения км/ч": [],
                "Плотность траффика": [],
                "Среднее своб. время сек": [],
                "Средняя задержка сек": [],
                "Временной индекс": [],
                "Время наблюдения сек": []
            }
            for period in sector.periods_data:
                stats["Интенсивность траффика"].append(traffic_intensity(
                    period.classwise_traveled_count,
                    self.size_coeffs,
                    period.observation_time
                ))

                vehicles_travel_time = period.ids_travel_time.values()
                vehicles_free_time = period.free_travel_time.values()
                stats["Среднее время проезда сек"].append(mean_travel_time(vehicles_travel_time)*SECS_IN_HOUR)
                stats["Средняя скорость движения км/ч"].append(mean_vehicle_speed(vehicles_travel_time, sector.length))

                stats["Плотность траффика"].append(traffic_density(
                    period.classwise_traveled_count,
                    self.size_coeffs,
                    vehicles_travel_time,
                    sector.length,
                    period.observation_time,
                    lane_count=sector.lanes_count
                ))

                stats["Среднее своб. время сек"].append(mean_free_time(
                    vehicles_free_time
                )*SECS_IN_HOUR)
                stats["Средняя задержка сек"].append(mean_vehicle_delay(
                    vehicles_travel_time,
                    vehicles_free_time
                )*SECS_IN_HOUR)
                stats["Временной индекс"].append(time_index(
                    vehicles_travel_time,
                    vehicles_free_time
                ))

                stats["Время наблюдения сек"].append(period.observation_time)
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

    def __get_vehicle_travel_time_debug(self, vehicle_id: int) -> float:
        # Get travel time for a vehicle by its ID
        for sector in self.sectors:
            if vehicle_id in sector.ids_start_time:
                return self.period_timer.unresettable_time - sector.ids_start_time[vehicle_id]
            elif vehicle_id in sector.ids_travel_time:
                return sector.ids_travel_time[vehicle_id]
        return None
