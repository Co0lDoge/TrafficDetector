import cv2
import numpy as np

def is_inside_zone(center, zone):
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0

class VehicleID:
    def __init__(self, class_name: str, bb):
        self.track_class = class_name
        self.bb = bb

class Region:
    def __init__(self, points):
        self.points = points
        self.counted_ids: dict[int, VehicleID] = {}

    def count_tracklet(self, box, track_id, track_class):
        bbox_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        crossed_before = track_id in self.counted_ids

        if not crossed_before and is_inside_zone(bbox_center, self.points):
            self.counted_ids[track_id] = VehicleID(track_class, box)
            
    def draw_regions(self, im0):
        for i in range(len(self.points)):
            cv2.line(
                im0,
                self.points[i],
                self.points[(i + 1) % len(self.points)],
                (0, 255, 0),
                thickness=2,
            )
                
