class Detector():
    def __init__(self, model, imgsize):
        self.model = model

        # Изменение размера изображения до кратного 32
        height, width = imgsize
        adjusted_width = (width + 32 - 1) // 32 * 32
        adjusted_height = (height + 32 - 1) // 32 * 32
        self.imgsize = (adjusted_height, adjusted_width)

    def track(
        self,
        frame: tuple,
    ):
        track_results = self.model.track(frame, persist=True, imgsz=self.imgsize)
        if track_results[0].boxes.id is not None:
            boxes = track_results[0].boxes.xyxy.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            classes = track_results[0].boxes.cls.cpu().tolist()
            return boxes, track_ids, classes
        else:
            return None
