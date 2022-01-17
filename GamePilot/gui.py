import PySimpleGUI as sg

THEME = "DarkGrey9"
sg.theme(THEME)


class Gui:
    slider_range = (-300, 300)
    hwnd = None

    # Settings =====
    process_name = "DrivingDemo"

    skew_top = 60
    expand_top = -70
    skew_bottom = 10
    expand_bottom = 60

    moving_avg = 3

    skewed_view_enable = False
    preprocessing_view_enable = False
    window_view_enable = False
    lines_view_enable = True

    forward_keybind = "Left Joystick Y-axis"
    steering_method = "Joystick"

    white_tresh = 200

    paused = True

    # Settings end =====

    layout = [
        [
            sg.Column([
                [
                    sg.Column([
                        [sg.Text("Press [Right Shift] to start", expand_x=True)]
                    ], justification="c")
                ],
                [
                    sg.Column([
                        [
                            sg.Text("FPS: 0", expand_x=True, key="fps"),
                            sg.Text("Curve Radius: 0", expand_x=True, key="curve_radius"),
                            sg.Text("Center Offset: 0 cm", expand_x=True, key="center_offset"),
                            sg.Text("Status: Idle", expand_x=True, key="status")
                        ]
                    ], justification="c")
                ],
                [sg.HSeparator(pad=(10, 20))],
                [
                    sg.Text(f"Game Process Name", expand_x=True),
                    sg.InputText(default_text=process_name, size=(30, 10),
                                 enable_events=True, key="process_name")
                ],
                [
                    sg.Text(f"Skew top ({skew_top})", expand_x=True),
                    sg.Slider(range=slider_range, default_value=skew_top, size=(50, 10), orientation="h",
                              enable_events=True, key="change_skew_top")
                ],
                [
                    sg.Text(f"Expand top ({expand_top})", expand_x=True),
                    sg.Slider(range=slider_range, default_value=expand_top, size=(50, 10), orientation="h",
                              enable_events=True, key="change_expand_top")
                ],
                [
                    sg.Text(f"Skew bottom ({skew_bottom})", expand_x=True),
                    sg.Slider(range=slider_range, default_value=skew_bottom, size=(50, 10), orientation="h",
                              enable_events=True, key="change_skew_bottom")
                ],
                [
                    sg.Text(f"Expand bottom ({expand_bottom})", expand_x=True),
                    sg.Slider(range=slider_range, default_value=expand_bottom, size=(50, 10), orientation="h",
                              enable_events=True, key="change_expand_bottom")
                ],
                [
                    sg.Text(f"Moving average length ({moving_avg})", expand_x=True),
                    sg.Slider(range=(1, 15), default_value=moving_avg, size=(50, 10), orientation="h",
                              enable_events=True, key="change_moving_avg")
                ],
                [
                    sg.Text(f"White Shade Threshold ({white_tresh})", expand_x=True),
                    sg.Slider(range=(0, 255), default_value=white_tresh, size=(50, 10), orientation="h",
                              enable_events=True, key="change_white_tresh")
                ],
                [
                    sg.Text(f"Forward keybind", expand_x=True),
                    sg.DropDown(values=["W", "R2", "Left Joystick Y-axis"], default_value=forward_keybind,
                                size=(20, 10), enable_events=True, key="forward_keybind")
                ],
                [
                    sg.Text(f"Steering Method", expand_x=True),
                    sg.DropDown(values=["Joystick", "ETS2 Mouse"], default_value=steering_method,
                                size=(20, 10), enable_events=True, key="steering_method")
                ],
                [sg.HSeparator(pad=(10, 20))],
                [
                    sg.Text(f"Live view", expand_x=True),
                    sg.Radio("Disable", 1, default=False, enable_events=True, key="view_disable"),
                    sg.Radio("Skewed", 1, default=False, enable_events=True,
                             key="skewed_view_enable"),
                    sg.Radio("Preprocessing", 1, default=False, enable_events=True,
                             key="preprocessing_view_enable"),
                    sg.Radio("Sliding Window", 1, default=False, enable_events=True,
                             key="window_view_enable"),
                    sg.Radio("Line Overlay", 1, default=True, enable_events=True, key="lines_view_enable")
                ],
            ], vertical_alignment="top"),
            sg.Column([
                [sg.Image(filename='', key='view_image', size=(500, 500))]
            ], key="image_container")
        ]
    ]

    def __init__(self):
        self.window = sg.Window("GamePilot", self.layout)

    def check_events(self, event, values):
        # I'm sorry.
        if event == "change_skew_top":
            self.skew_top = values["change_skew_top"]
        elif event == "change_expand_top":
            self.expand_top = values["change_expand_top"]
        elif event == "change_skew_bottom":
            self.skew_bottom = values["change_skew_bottom"]
        elif event == "change_expand_bottom":
            self.expand_bottom = values["change_expand_bottom"]
        elif event == "change_moving_avg":
            self.moving_avg = values["change_moving_avg"]
        elif event == "change_white_tresh":
            self.white_tresh = values["change_white_tresh"]
        elif event == "forward_keybind":
            self.forward_keybind = values["forward_keybind"]
        elif event == "steering_method":
            self.steering_method = values["steering_method"]
        elif event == "process_name":
            self.process_name = values["process_name"]
        elif event == "status":
            self.status = values["status"]
        elif event == "skewed_view_enable" \
                or event == "preprocessing_view_enable" \
                or event == "lines_view_enable" \
                or event == "window_view_enable":
            self.window["view_image"].update(visible=True)
            self.skewed_view_enable = values["skewed_view_enable"]
            self.preprocessing_view_enable = values["preprocessing_view_enable"]
            self.window_view_enable = values["window_view_enable"]
            self.lines_view_enable = values["lines_view_enable"]
        elif event == "view_disable":
            self.window["view_image"].update(visible=False)


gui = Gui()
