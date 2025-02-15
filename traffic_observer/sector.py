from typing import Sequence, Final, List

import pandas as pd
from ultralytics.solutions import ObjectCounter

from funcs import *
from traffic_observer.period import Period
from traffic_observer.step_timer import StepTimer

Hour = float
Secs = float
Kilometer = float
SECS_IN_HOUR: Final = 3600


class Sector:
    def __init__(self, vehicle_classes):
        self.periods_data: List[Period] = []
        self.ids_travel_time = {}
        self.classwise_traveled_count = {cls: 0 for cls in vehicle_classes}
        self.ids_start_time = {}
        self.ids_blacklist = set()
        self.travelling_ids = {}


class SectorManager:
    def __init__(
            self,
            length: int,  # Длина сектора в километрах
            lane_count: int,
            vehicle_classes: Sequence[str],
            time_step: int,
            observation_time: Secs,
            vechicle_size_coeffs: dict[str, float],
            region_count: int
    ):
        self.size_coeffs = vechicle_size_coeffs
        self.vehicle_classes = vehicle_classes
        self.length = length
        self.lane_count = lane_count
        self.len_sector = region_count // 2  # Сектор - пространство между двумя регионами

        self.observation_period: Secs = observation_time
        self.period_timer = StepTimer(time_step)

        self.sectors = [Sector(self.vehicle_classes) for _ in range(self.len_sector)]

    def update(self, regions: List[ObjectCounter]):
        self.period_timer.step_forward()
        if self.period_timer.time >= self.observation_period:
            self.new_period()

        # Итерация по секторам и регионам, каждому сектору соответствует два региона
        iter_region = iter(regions)
        iter_sector = iter(self.sectors)
        for _ in range(self.len_sector):
            start_counter = next(iter_region)
            end_counter = next(iter_region)
            sector = next(iter_sector)
            #print(sector.classwise_traveled_count)

            for vid in start_counter.counted_ids:
                if vid not in sector.ids_start_time and vid not in sector.ids_blacklist:
                    sector.ids_start_time[vid] = self.period_timer.unresettable_time

            for vid in end_counter.counted_ids:
                try:
                    if vid not in sector.ids_blacklist:
                        dt = self.period_timer.unresettable_time - sector.ids_start_time[vid]
                        sector.ids_start_time.pop(vid)

                        sector.ids_travel_time[vid] = dt

                        cls_name = end_counter.counted_ids[vid].cls_name
                        sector.classwise_traveled_count[cls_name] += 1
                        sector.ids_blacklist.add(vid)
                except KeyError:
                    if vid in sector.ids_blacklist:
                        sector.ids_blacklist.remove(vid)

    def new_period(self):
        for sector in self.sectors:
            sector.periods_data.append(Period(
                sector.ids_travel_time.copy(),
                sector.classwise_traveled_count.copy(),
                self.period_timer.time
            ))

            sector.ids_travel_time.clear()
            sector.classwise_traveled_count = {cls: 0 for cls in self.vehicle_classes}
        self.period_timer.reset()

    def traffic_stats(self) -> List[pd.DataFrame]:

        dataframes = []
        # TODO: Максим: сделать генерацию отчета за все периоды
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
        # TODO: Максим: сделать генерацию отчета за все периоды
        for sector in self.sectors:
            stats = {cls: [] for cls in self.vehicle_classes}
            for period in sector.periods_data:
                for cls in self.vehicle_classes:
                    stats[cls].append(period.classwise_traveled_count[cls])
            dataframes.append(pd.DataFrame(stats))

        return dataframes
