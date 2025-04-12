from directional_detector import directional_detect

directional_detect(
    video_path = "__test/video/galle_cross_quickstart.mp4",
    model_path = "__test/model/detector_yolov10s.pt",
    output_path = "__test/output/output_video.mp4",
    report_path = "__test/output/report.json",
    sector_path = "__test/region/multiregion_galle.json",
    show_image=True
)