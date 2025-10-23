import argparse
import logging
import os
import time
from pathlib import Path

import cv2


def get_videos(root_dir, recursive=False):
    root_dir = Path(root_dir)
    pattern = "**/*" if recursive else "*"
    for path in root_dir.glob(pattern):
        if path.is_file():
            yield path
        else:
            logging.warning(f"'{str(path)}' is not a file")

def is_blur_image(image, blur_threshold):
    if blur_threshold is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        l_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if l_var < blur_threshold:
            return l_var, True
    return 0, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from directory of videos")

    parser.add_argument(
        "video_dir",
        type=str,
        help="path to video directory",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="path to save result images. default is ./results",
        default="./results"
    )

    parser.add_argument(
        "-e", "--extension",
        type=str,
        help="extension of image to save in png or jpg. default is jpg",
        choices=["jpg", "png"],
        default="jpg"
    )

    parser.add_argument(
        "-f", "--flat",
        help="save all images from all videos in sigle directory <output_dir>/<image> (normally images will be save in <output_dir>/<video_name>/<image>)",
        action="store_true"
    )

    parser.add_argument(
        "-r", "--recursive",
        help="recursively get all videos under given <video_dir>",
        action="store_true"
    )

    parser.add_argument(
        "-fb", "--filter-blur-image",
        nargs="?",
        const=100,
        type=int,
        help="filter blur image using variance of laplacian (smaller the value, more blurry the image). default is 100 if used without value",
    )

    parser.add_argument(
        "-sf", "--skip-frame",
        nargs="?",
        const=1,
        type=int,
        help="skip saving image from video every N frame(s). default is 1 frame",
    )

    parser.add_argument(
        "-st", "--skip-time",
        nargs="?",
        const=1000,
        type=int,
        help="skip saving image from video every N milisecond. default is 1000 ms (1 sec)",
    )

    parser.add_argument(
        "-v", "--verbose",
        help="enable verbose mode",
        action="store_true"
    )

    args = parser.parse_args()

    # Validate
    if not os.path.isdir(args.video_dir):
        raise Exception(f"'{args.video_dir}' is not exist as directory")

    if args.skip_time is not None and args.skip_frame is not None:
        raise Exception(f"You can choose either to skip frame or time, not both")

    # Initialize
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    for vid_path in get_videos(args.video_dir, args.recursive):

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            logging.warning(f"'{str(vid_path)}' is not support video format")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Process '{str(vid_path)}' | FPS: {fps}")

        frame_count = 0
        try:
            while True:

                if args.skip_time is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, frame_count * args.skip_time) # <- this is very slow

                success, image = cap.read()
                if not success:
                    break

                if args.skip_frame is not None and frame_count % args.skip_frame != 0: 
                    frame_count+=1
                    continue

                l_var, is_blur = is_blur_image(image, args.filter_blur_image)
                if is_blur:
                    logging.info(f"Skip frame {frame_count} since laplacian variance is lower than the threshold ( {l_var} < {args.filter_blur_image} ) [blurry frame]")
                    frame_count+=1
                    continue

                vid_name = vid_path.stem
                image_name = f"{time.time()}-{frame_count}.{args.extension}"
                save_path = ""
                if args.flat:
                    save_path = os.path.join(args.output, f"{vid_name}-{image_name}")
                else:
                    save_dir = os.path.join(args.output, vid_name) 
                    Path(save_dir).mkdir(parents=True, exist_ok=True)

                    save_path = os.path.join(save_dir, image_name)

                cv2.imwrite(save_path, image)
                logging.info(f"Saved '{save_path}'")

                frame_count+=1

        except Exception as e:
            raise e
        finally:
            cap.release()

        logging.info(f"Finish process '{str(vid_path)}'")




