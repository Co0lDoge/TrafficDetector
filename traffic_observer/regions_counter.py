import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def is_inside_zone(center, zone):
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0


def _annotate(im0, annotator, box, track_id, cls):
    annotator.box_label(box, "", color=(255, 0, 0))
    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center


class VehicleID:
    def __init__(self, cls_name: str, bb):
        self.cls_name = cls_name
        self.bb = bb


class Region:
    def __init__(self, points, class_names):
        self.points = points
        self.classwise_count = {cls_name:0 for cls_name in class_names}
        self.counted_ids: dict[int, VehicleID] = {}


class RegionsCounter:
    def __init__(self, model, imgsz, regions_points: list[list[int]]):
        self.model = YOLO(model)
        self.regions = [Region(points, self.model.names.values()) for points in regions_points]
        width = imgsz[1]
        height = imgsz[0]
        adjusted_width = (width + 32 - 1) // 32 * 32
        adjusted_height = (height + 32 - 1) // 32 * 32
        self.imgsz = (adjusted_height, adjusted_width)
    
    def count(self, im0, *, annotate=False, draw_regions=True):
        results = self.model.track(im0, persist=True, imgsz=self.imgsz)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            if annotate:
                annotator = Annotator(im0, line_width=1, example=str(self.model.names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if annotate:
                    _annotate(im0, annotator, box, track_id, cls)

                for region in self.regions:
                    bbox_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    crossed_before = track_id in region.counted_ids
                    cls_name = self.model.names[cls]

                    if is_inside_zone(bbox_center, region.points) and not crossed_before:
                        region.classwise_count[cls_name] += 1
                        region.counted_ids[track_id] = VehicleID(cls_name, box)
                    elif crossed_before:
                        # TODO: Зачем это нужно? На результат это не влияет
                        region.counted_ids.pop(track_id, None)

        if draw_regions:
            for region in self.regions:
                for i in range(len(region.points)):
                    cv2.line(
                        im0,
                        region.points[i],
                        region.points[(i + 1) % len(region.points)],
                        (0, 255, 0),
                        thickness=2,
                    )
