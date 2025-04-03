import cv2
import numpy as np

def is_inside_zone(center, zone):
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0

class Lane:
    def __init__(self, points):
        self.points = points
        self.delay = 0
        self.counted_ids = set()

    def count_tracklet(self, box, track_id):
        bbox_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        crossed_before = track_id in self.counted_ids

        if not crossed_before and is_inside_zone(bbox_center, self.points):
            self.counted_ids.add(track_id)
            
    def draw_lane(self, im0):
        for i in range(len(self.points)):
            cv2.putText(im0, f"{self.delay:.2f}", 
                org=self.points[3], 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5,
                color=(255,255,255),
                thickness=1,
                lineType=2)
            cv2.line(
                im0,
                self.points[i],
                self.points[(i + 1) % len(self.points)],
                (120, 0, 255),
                thickness=2,
            )