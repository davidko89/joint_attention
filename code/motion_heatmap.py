import numpy as np
import cv2
import copy
from pathlib import Path

# PROJECT_PATH = Path(__file__).parents[1]
# DATA_PATH = Path(PROJECT_PATH, "data")
# VIDEO_PATH = (DATA_PATH, "raw/ASD/B701/B701_RJA_high_BL.MP4")


def motion_heatmap_code():
    video = "/home/cko4/project_joint/data/raw/ASD/B701/B701_RJA_high_BL.MP4"
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    child_id = video.split("/")[-2]
    task_name = video.split("/")[-1].split("_")[1]
    direction = video.split("/")[-1].split("_")[3].split(".")[0]

    out = cv2.VideoWriter(
        f"{child_id}_{task_name}_high_{direction}_output.mp4",
        fourcc,
        fps,
        (int(width), int(height)),
    )

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    num_frames = 100

    first_iteration_indicator = 1
    for i in range(0, num_frames):

        if first_iteration_indicator == 1:
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            width, height = gray.shape[:2]
            accum_image = np.zeros((width, height), np.uint8)
            first_iteration_indicator = 0
        else:
            ret, frame = cap.read()  # read a frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background

            # for testing purposes, show the result of the background subtraction
            # cv2.imshow('diff-bkgnd-frame', fgmask)

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
            # for testing purposes, show the threshold image
            # cv2.imwrite('diff-th1.jpg', th1)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            # for testing purposes, show the accumulated image
            # cv2.imwrite('diff-accum.jpg', accum_image)

            # for testing purposes, control frame by frame
            # raw_input("press any key to continue")

        # for testing purposes, show the current frame
        # cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    color_image = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    cv2.imwrite("diff-overlay.jpg", result_overlay)

    out.write(result_overlay)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    motion_heatmap_code()
