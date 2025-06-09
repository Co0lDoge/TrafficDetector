import subprocess

def run_video_processing():
    command = [
        '.venv/Scripts/python.exe', 'sector_detector.py', 
        '--video-path', '__test/video/test_720p.mp4', 
        '--model-path', '__test/model/legacy_model.pt', 
        '--output-path', '__test/output/test_720p_output.mp4', 
        '--report-path', '__test/output/test_720p_report.xlsx', 
        '--sector-path', '__test/region/test_720p.json'
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Cannot launch: {e}")

if __name__ == "__main__":
    run_video_processing()