import keyboard as kb
import vgamepad as vg
import win32api
import win32con

from GamePilot.gui import gui


class Steering:
    angle = 0

    # Mouse steering
    angle_avg = []
    prev_angle = 0

    def __init__(self):
        self.gp = vg.VX360Gamepad()

    def calc_angle(self, curve_radius, center_offset):
        center_offset_range = 100
        center_offset = max(-center_offset_range, center_offset)
        center_offset = min(center_offset_range, center_offset)
        offset_angle = int(-center_offset * (32767 / center_offset_range))

        # TODO: Use curve_radius for steering prediction

        if gui.steering_method == "Joystick":
            self.set_steer(offset_angle)
        elif gui.steering_method == "ETS2 Mouse":
            self.angle_avg.append(offset_angle / 200)
            if len(self.angle_avg) > 3:
                self.angle_avg.pop(0)
            angle = sum(self.angle_avg) / len(self.angle_avg)
            self.set_mouse_steer(angle)

    def set_steer(self, val: float):
        if val < -32768 or val > 32767:
            raise Exception("Setting value out of bounds")
        self.angle = int(val)
        self.update_keys()

    def set_mouse_steer(self, val: float):
        self.angle = int(val)
        self.update_keys()

    def update_keys(self):
        y_axis = 0
        if gui.forward_keybind == "W" and not kb.is_pressed("w"):
            kb.press("w")
        elif gui.forward_keybind == "R2":
            self.gp.right_trigger(value=255)
        elif gui.forward_keybind == "Left Joystick Y-axis":
            y_axis = 32767
            if not gui.steering_method == "Joystick":
                self.gp.left_joystick(x_value=0, y_value=y_axis)
                self.gp.update()

        if gui.steering_method == "Joystick":
            self.gp.left_joystick(x_value=self.angle, y_value=y_axis)
            self.gp.update()
        elif gui.steering_method == "ETS2 Mouse":
            # TODO: works, but its not great
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, (self.angle - self.prev_angle), 0, 0, 0)
            self.prev_angle = self.angle
            pass


steering = Steering()
