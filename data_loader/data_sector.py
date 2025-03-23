class DataSector:
    def __init__(self, sector_id, start_points, end_points, lanes_points, lanes_count, sector_length, max_speed):
        self.id = sector_id
        self.start_points = start_points
        self.end_points = end_points
        self.lanes_points = lanes_points
        self.lanes_count = lanes_count
        self.sector_length = sector_length
        self.max_speed = max_speed
    