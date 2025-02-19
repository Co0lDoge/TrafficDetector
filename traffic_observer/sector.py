from typing import Sequence, List

import pandas as pd
import cv2
import logging

from funcs import *
from traffic_observer.period import Period
from traffic_observer.step_timer import StepTimer
from traffic_observer.regions_counter import RegionCounter

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
            length: int,  # Длина сектора в километрах
            lane_count: int,
            max_speed: int,
            vehicle_classes: Sequence[str],
            time_step: int,
            observation_time: int, # Время наблюдения в секундах
            vechicle_size_coeffs: dict[str, float],
            region_counter: RegionCounter
    ):
        self.size_coeffs = vechicle_size_coeffs
        self.vehicle_classes = vehicle_classes
        self.length = length
        self.lane_count = lane_count
        self.num_sectors = len(region_counter.regions)//2  # Каждому сектору соответствует два региона
        self.region_counter = region_counter
        self.observation_period = observation_time
        self.period_timer = StepTimer(time_step)

        self.sectors = [Sector(self.vehicle_classes) for _ in range(self.num_sectors)]

    def update(self, frame: cv2.typing.MatLike):
        self.region_counter.count(frame, annotate=True)
        logging.info(f"Обработан кадр по времени {self.period_timer.time}")

        self.period_timer.step_forward()
        if self.period_timer.time >= self.observation_period:
            self.new_period()

        # Итерация по секторам и регионам, каждому сектору соответствует два региона
        for i in range(self.num_sectors):
            start_counter = self.region_counter.regions[2 * i]
            end_counter = self.region_counter.regions[2 * i + 1]
            sector = self.sectors[i]

            # Сохранение времени проезда для всего транспорта, перешедшего конечный регион
            for vid in start_counter.counted_ids:
                if vid not in sector.ids_start_time and vid not in sector.ids_blacklist:
                    sector.ids_start_time[vid] = self.period_timer.unresettable_time

            # Обновление времени начала движения для всего транспорта перешедшего начальный регион
            for vid in end_counter.counted_ids:
                if vid not in sector.ids_blacklist and vid in sector.ids_start_time:
                    dt = self.period_timer.unresettable_time - sector.ids_start_time[vid]
                    sector.ids_start_time.pop(vid)

                    sector.ids_travel_time[vid] = dt

                    class_name = end_counter.counted_ids[vid].class_name
                    sector.classwise_traveled_count[class_name] += 1
                    sector.ids_blacklist.add(vid)
                    
            
        logging.info(f"Обновлены сектора по времени {self.period_timer.time}")

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
