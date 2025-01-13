import argparse
from ffmpeg import FFmpeg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--use-nvenc", action="store_true")

    args = parser.parse_args()

    codec = "libx264"
    if args.use_nvenc:
        codec = "h264_nvenc"

    # TODO опция -y: необходимо или нет?
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(args.input_file)
        .output(
            args.output_file,
            {"codec:v": codec},
        )
    )

    ffmpeg.execute()
