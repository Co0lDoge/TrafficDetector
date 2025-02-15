class Period:
    def __init__(self, ids_travel_time, classwise_traveled_count, observation_time):
        self.ids_travel_time = ids_travel_time
        self.classwise_traveled_count = classwise_traveled_count
        
        # Нужно чтобы использовать время из таймера, так как могло пройти меньше времени, чем observation-time
        self.observation_time = observation_time 