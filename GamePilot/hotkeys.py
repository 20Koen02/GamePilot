from pynput import keyboard
import keyboard as kb


from GamePilot.gui import gui
from GamePilot.steering import steering


def on_press(key):
    if key == keyboard.Key.shift_r:
        gui.paused = not gui.paused

        kb.release("w")
        steering.gp.left_joystick(x_value=0, y_value=0)
        steering.gp.right_trigger(value=0)
        steering.gp.update()
