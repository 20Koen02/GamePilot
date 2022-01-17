import time

import numpy as np
import win32gui
from mss import mss

from GamePilot.gui import gui
from GamePilot.lane_detection import LaneDetection


def main():
    for t in range(3):
        print(f"Generating images in... {3 - t}")
        time.sleep(1)

    hwnd = win32gui.FindWindow(None, "DrivingDemo")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    game_window = {'top': top, 'left': left, 'width': right - left, 'height': bottom - top}
    with mss() as sct:
        game_capture = np.array(sct.grab(game_window))
        LaneDetection(game_capture, True)


if __name__ == '__main__':
    main()
