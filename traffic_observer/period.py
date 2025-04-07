class Period:
    def __init__(self, ids_travel_time, classwise_traveled_count, free_travel_time, observation_time):
        # TODO: set type hints
        self.ids_travel_time = ids_travel_time
        self.classwise_traveled_count = classwise_traveled_count
        self.free_travel_time = free_travel_time
        
        # Нужно чтобы использовать время из таймера, так как могло пройти меньше времени, чем observation-time
        self.observation_time = observation_time 