import time
from ctypes import windll

import PySimpleGUI as sg
import numpy as np
import pywintypes
import win32gui
from mss import mss
from pynput import keyboard

from GamePilot.gui import gui
from GamePilot.hotkeys import on_press
from GamePilot.lane_detection import LaneDetection

windll.user32.SetProcessDPIAware()


def main():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mss() as sct:
        once = True
        prev_frame_time = 0
        while True:
            event, values = gui.window.read(0)
            if once:
                gui.window.move(-1820, 100)
                event, values = gui.window.read(0)
                time.sleep(0.1)
            if event == sg.WIN_CLOSED:
                return
            if event != "__TIMEOUT__":
                gui.check_events(event, values)

            while not gui.hwnd:
                event, values = gui.window.read(0)
                if event == sg.WIN_CLOSED:
                    return
                if event != "__TIMEOUT__":
                    gui.check_events(event, values)
                try:
                    gui.hwnd = win32gui.FindWindow(None, gui.process_name)
                    if gui.hwnd:
                        break
                except pywintypes.error:
                    pass
                gui.window["status"].update(value='Status: Window not found')

            once = False
            if gui.paused:
                gui.window["status"].update(value="Status: Paused")
                continue
            else:
                gui.window["status"].update(value="Status: Running")

            try:
                left, top, right, bottom = win32gui.GetWindowRect(gui.hwnd)
            except pywintypes.error:
                gui.window["status"].update(value='Status: Window not found')
                return


            game_window = {'top': top, 'left': left, 'width': right - left, 'height': bottom - top}
            game_capture = np.array(sct.grab(game_window))
            LaneDetection(game_capture)

            # FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            gui.window["fps"].update(value='FPS: ' + str(round(fps)))

        gui.window.close()


if __name__ == '__main__':
    main()
