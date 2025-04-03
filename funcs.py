import statistics
from typing import Iterable
from traffic_observer.lane import Lane

Hours = float
Percents = float
Seconds = float
Kilometer = float
Kmph = float
SECS_IN_HOUR = 3600


def traffic_intensity(
    classwise_traveled_count: dict[str, Seconds],
    vehicle_size_coeffs: dict[str, float],
    observation_time: Seconds|float,
) -> float:
    ''' Интенсивность движения транспортных средств (1 пункт) '''

    s = 0
    for cls_name, traveled_count in classwise_traveled_count.items():
        s += traveled_count * vehicle_size_coeffs.get(cls_name, 1)

    return s / (observation_time / SECS_IN_HOUR)
    

def vehicle_class_share(
    cls_name: str, classwise_traveled_count: dict[str, int],
) -> Percents:
    ''' Доля транспортных средств каждой рассчетной категории (2 пункт) '''

    total = sum(classwise_traveled_count.values())
    return 100 * classwise_traveled_count[cls_name] / total

def mean_travel_time(vehicles_travel_time: Iterable[Seconds|float]) -> Hours:
    ''' Среднее время движения транспортных средств (3 пункт) (probably redundant) '''

    try:
        return statistics.mean(vehicles_travel_time) / SECS_IN_HOUR
    except statistics.StatisticsError:
        return float("nan")


def mean_vehicle_speed(
    vehicles_travel_time: Iterable[Seconds|float],
    sector_length: Kilometer,
) -> Kmph:
    ''' Средняя скорость движения транспортных средств (3 пункт) '''

    return sector_length / mean_travel_time(vehicles_travel_time)

def mean_free_time(vehicles_free_time: Iterable[Seconds|float]) -> Hours:
    try:
        return statistics.mean(vehicles_free_time) / SECS_IN_HOUR
    except statistics.StatisticsError:
        return float("nan")

def mean_vehicle_delay(
    vehicles_travel_time: list[list[float]],
    vehicles_free_time: list[int]
) -> Hours:
    return mean_travel_time(vehicles_travel_time) - mean_free_time(vehicles_free_time)
    
def time_index(
    vehicles_travel_time: list[list[float]],
    vehicles_free_time: list[int]
):
    free_time = mean_free_time(vehicles_free_time)
    if free_time != 0:
        return mean_travel_time(vehicles_travel_time)/mean_free_time(vehicles_free_time)
    else:
        return float("nan")

def traffic_density(
    classwise_traveled_count: dict[str, Seconds],
    vehicle_size_coeffs: dict[str, float],
    vehicles_travel_time: Iterable[Seconds|float],
    sector_length: Kilometer,
    observation_time: Seconds|float,
    lane_count: int,
):
    ti = traffic_intensity(classwise_traveled_count, vehicle_size_coeffs, observation_time)
    v = mean_vehicle_speed(vehicles_travel_time, sector_length)

    return ti / (lane_count * v)
