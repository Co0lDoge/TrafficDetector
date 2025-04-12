import argparse

def load_args():
    # Добавление аргументов запуска
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True, help="Путь к видео")
    parser.add_argument("--model-path", type=str, required=True, help="Путь к модельке")
    parser.add_argument("--output-path", type=str, required=True, help="Путь для выходного файлы")
    parser.add_argument("--report-path", type=str, required=True, help="Путь для выходного отчета")
    parser.add_argument("--sector_path", type=str, required=True, help="Массив точек областей")

    # Получение всех аргументов
    args = parser.parse_args()

    # Доступ к аргументам
    video_path = args.video_path
    model_path = args.model_path
    output_path = args.output_path
    report_path = args.report_path
    sector_path = args.sector_path
    
    return video_path, model_path, output_path, report_path, sector_path