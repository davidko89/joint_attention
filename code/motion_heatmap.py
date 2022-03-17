import numpy as np
import cv2
import copy
from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
video = Path(DATA_PATH, "raw/ASD/B701/B701_RJA_high_BL.MP4")


def motion_heatmap_code():
    cap = cv2.VideoCapture(str(video))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(
        "B701_RJA_high_output.mp4", fourcc, fps, (int(height), int(width))
    )

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    num_frames = 300

    first_iteration_indicator = 1
    for i in range(0, num_frames):
        if first_iteration_indicator == 1:
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fgmask = fgbg.apply(gray)

            cv2.imshow("diff-bkgnd-frame", fgmask)

            # If you want motion to be picked up more, increase maxValue
            # To pick up the least amount of motion, set maxValue = 1
            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)

            accum_image = cv2.add(accum_image, th1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

    # overlay the color mapped image to the first frame
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final overlay image
    cv2.imwrite("diff-overlay.jpg", result_overlay)

    out.write(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    motion_heatmap_code()
