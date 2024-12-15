import time
from typing import Sequence, Final

import pandas as pd
from ultralytics.solutions import ObjectCounter

from funcs import *

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
        observation_time: Secs = 3600,
        vechicle_size_coeffs: dict[str, float] = None,
    ):
        self.size_coeffs = vechicle_size_coeffs
        if vechicle_size_coeffs is None:
            self.size_coeffs = {
                "car": 1,
                "motorbike": 0.5,
                "truck": 1.8,
                "road_train": 2.7,
                "bus": 2.2,
            }

        self.vehicle_classes = vehicle_classes
        self.length = length
        self.lane_count = lane_count

        self.observation_period: Secs = observation_time
        self.periods_data = []

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
        self.periods_data.append([
            self.ids_travel_time.copy(),
            self.classwise_traveled_count.copy(),
        ])

        self.period_timer.reset()
        self.ids_travel_time.clear()
        self.classwise_traveled_count = {cls:0 for cls in self.vehicle_classes}

    def traffic_stats(self) -> pd.DataFrame:
        stats = {
            "Интенсивность траффика": [],
            "Среднее время проезда": [],
            "Средняя скорость движения": [],
            "Плотность траффика": []
        }
        for ids_travel_time, classwise_traveled_count in self.periods_data:
            stats["Интенсивность траффика"].append(traffic_intensity(
                classwise_traveled_count,
                self.size_coeffs,
                self.observation_period
            ))

            vehicles_travel_time = ids_travel_time.values()
            stats["Среднее время проезда"].append(mean_travel_time(vehicles_travel_time))
            stats["Средняя скорость движения"].append(mean_vehicle_speed(vehicles_travel_time, self.length))

            stats["Плотность траффика"].append(traffic_density(
                classwise_traveled_count,
                self.size_coeffs,
                vehicles_travel_time,
                self.length,
                self.observation_period
            ))

        return pd.DataFrame(stats)
