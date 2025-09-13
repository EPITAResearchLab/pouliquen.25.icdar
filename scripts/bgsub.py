from __future__ import print_function
from pathlib import Path
import cv2 as cv
import numpy as np
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(
        description="This program performs background subtraction and OVD enhancement.")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a sequence of images.",
        default="./images",
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Background subtraction method (MEAN, MEDIAN).",
        default="MEDIAN",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("processing", args.input)
    frames_paths = sorted(glob(args.input + "/*.jpg"))
    frames = [cv.imread(f) for f in frames_paths if cv.imread(f) is not None]

    if not frames:
        print(f"No valid frames found for {args.input}")
        return

    outpath = args.input.replace("/cropped/", "/cropped_sub/")
    if outpath != args.input:
        print("Creating output directory", outpath)
        Path(outpath).mkdir(exist_ok=True, parents=True)
    else:
        print(f"The input and output path are similar {args.input}")
        return

    # Compute background
    if args.algo == "MEAN":
        print("Computing the mean frame...")
        background_frame = np.mean(frames, axis=0).astype(np.uint8)
    elif args.algo == "MEDIAN":
        print("Computing the median frame...")
        background_frame = np.median(frames, axis=0).astype(np.uint8)
    else:
        print("Invalid algorithm specified. Use MEAN or MEDIAN.")
        return

    max_sv_diff = 0

    # enhance signal
    sv_differences = []
    frames_diff = []
    for i, frame in enumerate(frames):
        if frame is None:
            break
        frame_diff = cv.absdiff(frame, background_frame)
        # Convert the current frame to HSV
        hsv_frame = cv.cvtColor(frame_diff, cv.COLOR_BGR2HSV)

        # Calculate the difference in terms of S*V after sub. with the background
        sv_diff = hsv_frame[..., 1].astype(float) * hsv_frame[..., 2].astype(float)

        sv_differences.append(sv_diff)
        frames_diff.append(frame_diff)

        max_sv_diff = max(max_sv_diff, np.max(sv_diff))

    # Normalize and display each frame's SV difference using the maximum SV difference found
    for i, sv_diff in enumerate(sv_differences):
        if frames[i] is None:
            break

        combined_final = frames_diff[i].copy().astype(float)
        for j in range(3):
            combined_final[..., j] *= sv_diff / max_sv_diff

        if outpath != args.input:
            cv.imwrite(frames_paths[i].replace("/cropped/", "/cropped_sub/"), combined_final)


if __name__ == "__main__":
    main()
