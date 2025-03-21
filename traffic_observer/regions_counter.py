import cv2
import numpy as np

def is_inside_zone(center, zone):
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0

class VehicleID:
    def __init__(self, class_name: str, bb):
        self.class_name = class_name
        self.bb = bb


class Region:
    def __init__(self, points, class_names):
        self.points = points
        self.classwise_count = {cls_name:0 for cls_name in class_names}
        self.counted_ids: dict[int, VehicleID] = {}


class RegionCounter:
    def __init__(self, class_names, regions_points: list[list[int]]):
        self.regions = [Region(points, class_names.values()) for points in regions_points]
        self.class_names = class_names
    
    def count_tracklet(self, box, track_id, track_class):
        for region in self.regions:
            bbox_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            crossed_before = track_id in region.counted_ids
            cls_name = self.class_names[track_class]

            if crossed_before:
                # Объект постоянно попадает из изчезает из зоны
                # TODO: придумать способ окончательно перестать отслеживать объект
                # region.counted_ids.pop(track_id, None)
                pass
            elif is_inside_zone(bbox_center, region.points):
                region.classwise_count[cls_name] += 1
                region.counted_ids[track_id] = VehicleID(cls_name, box)

    def draw_regions(self, im0):
        for region in self.regions:
                for i in range(len(region.points)):
                    cv2.line(
                        im0,
                        region.points[i],
                        region.points[(i + 1) % len(region.points)],
                        (0, 255, 0),
                        thickness=2,
                    )
            
