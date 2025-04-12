import cv2
import numpy as np

def is_inside_zone(center, zone):
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0

class Region:
    def __init__(self, points):
        self.points = points

    def is_intersected(self, box):
        bbox_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        return is_inside_zone(bbox_center, self.points)
            
    def draw_regions(self, im0, color = (0, 255, 0), text: str = None):
        for i in range(len(self.points)):
            cv2.line(
                im0,
                self.points[i],
                self.points[(i + 1) % len(self.points)],
                color,
                thickness=2,
            )
        
        if text:
            center_x = int(sum(point[0] for point in self.points) / len(self.points))
            center_y = int(sum(point[1] for point in self.points) / len(self.points))
            
            cv2.putText(
                im0,
                text,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                thickness=2,
                lineType=cv2.LINE_4
            )
                
