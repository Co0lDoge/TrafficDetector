import time
from typing import Sequence, Final, List

import pandas as pd
from ultralytics.solutions import ObjectCounter

from funcs import *
from period import Period

Hour = float
Secs = float
Kilometer = float
SECS_IN_HOUR: Final = 3600


class Sector:
    def __init__(
        self,
        length: Kilometer,
        lane_count: int,
        vehicle_classes: Sequence[str],
        timer,
        observation_time: Secs,
        vechicle_size_coeffs: dict[str, float],
    ):
        self.size_coeffs = vechicle_size_coeffs
        self.vehicle_classes = vehicle_classes
        self.length = length
        self.lane_count = lane_count

        self.observation_period: Secs = observation_time
        self.periods_data: List[Period] = []

        self.period_timer = timer
        self.ids_travel_time = {}
        self.classwise_traveled_count = {cls:0 for cls in self.vehicle_classes}
        self.ids_start_time = {}
        self.ids_blacklist = set()
        self.travelling_ids = {}

    def update(self, start_counter: ObjectCounter, end_counter: ObjectCounter):
        self.period_timer.step_forward()
        if self.period_timer.time >= self.observation_period:
            self.new_period()

        for vid in start_counter.counted_ids:
            if vid not in self.ids_start_time and vid not in self.ids_blacklist:
                self.ids_start_time[vid] = self.period_timer.unresettable_time    # TODO да простит меня бог

        for vid in end_counter.counted_ids:
            try:
                if vid not in self.ids_blacklist:
                    dt = self.period_timer.unresettable_time - self.ids_start_time[vid]
                    self.ids_start_time.pop(vid)

                    self.ids_travel_time[vid] = dt

                    cls_name = end_counter.counted_ids[vid].cls_name
                    self.classwise_traveled_count[cls_name] += 1
                    self.ids_blacklist.add(vid)
            except KeyError:
                if vid in self.ids_blacklist:
                    self.ids_blacklist.remove(vid)

    def new_period(self):
        self.periods_data.append(Period(
            self.ids_travel_time.copy(),
            self.classwise_traveled_count.copy(),
            self.period_timer.time
        ))

        self.period_timer.reset()
        self.ids_travel_time.clear()
        self.classwise_traveled_count = {cls:0 for cls in self.vehicle_classes}

    def traffic_stats(self) -> pd.DataFrame:
        stats = {
            "Интенсивность траффика": [],
            "Среднее время проезда": [],
            "Средняя скорость движения": [],
            "Плотность траффика": [],
            "Время наблюдения": []
        }
        for period in self.periods_data:
            stats["Интенсивность траффика"].append(traffic_intensity(
                period.classwise_traveled_count,
                self.size_coeffs,
                self.observation_period
            ))

            vehicles_travel_time = period.ids_travel_time.values()
            stats["Среднее время проезда"].append(mean_travel_time(vehicles_travel_time))
            stats["Средняя скорость движения"].append(mean_vehicle_speed(vehicles_travel_time, self.length))

            stats["Плотность траффика"].append(traffic_density(
                period.classwise_traveled_count,
                self.size_coeffs,
                vehicles_travel_time,
                self.length,
                self.observation_period,
                lane_count=self.lane_count
            ))

            stats["Время наблюдения"].append(period.observation_time)

        return pd.DataFrame(stats)
    
    def classwise_stats(self) -> pd.DataFrame:
        stats = {cls: [] for cls in self.vehicle_classes}
        for period in self.periods_data:
            for cls in self.vehicle_classes:
                stats[cls].append(period.classwise_traveled_count[cls])

        return pd.DataFrame(stats)
