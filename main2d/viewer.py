import OpenGL.GL as GL
import glfw
import numpy as np
from itertools import cycle
from tostudents.libs.transform import Trackball, translate, scale
import imgui
from imgui.integrations.glfw import GlfwRenderer
from tostudents.shape2d.shape2d import *
from tostudents.main.axes import Axes


class UIState:
    def __init__(self):
        self.render_modes = ["Flat", "Texture", "Gouraud", "Phong", "Wireframe"]
        self.render_mode_idx = 0

        # Chỉ 2D shapes
        self.shapes_2d = [
            "Triangle2D", "Rectangle2D", "Pentagon2D", "Hexagon2D",
            "Circle2D", "Ellipse2D", "Trapezoid2D", "Star2D", "Arrow2D"
        ]
        self.shape_idx = 0

        # Transform
        self.translate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]

        # Toggle
        self.show_axes = True

        #color
        self.color = [1.0, 1.0, 1.0]

    def reset_transform(self):
        self.translate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]


class Viewer:
    def __init__(self, width=1200, height=900):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        self.width = width
        self.height = height

        # --- Setup GLFW ---
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, '2D Shape Viewer', None, None)
        glfw.make_context_current(self.win)

        # --- ImGui ---
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)
        self.make_colors()

        # --- Trackball ---
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # --- State ---
        self.state = UIState()
        self.drawables = []
        self._managed_drawable = None

        # --- Axes ---
        self.axes = Axes(
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main2d/gouraud.vert",
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main2d/gouraud.frag",
            length=2.0
        ).setup()

        # --- Callbacks ---
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

    def make_colors(self):
        style = imgui.get_style()
        style.window_rounding = 4.0
        colors = style.colors
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.12, 0.16, 0.20, 1.0)
        colors[imgui.COLOR_BUTTON] = (0.2, 0.48, 0.54, 1.0)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.3, 0.6, 0.7, 1.0)

    def render_ui(self, state):
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(320, 600)
        imgui.begin("Control Panel", True)

        imgui.text("Render Mode:")
        _, state.render_mode_idx = imgui.combo("##render_mode", state.render_mode_idx, state.render_modes)

        imgui.text("2D Shape:")
        _, state.shape_idx = imgui.combo("##shape2d", state.shape_idx, state.shapes_2d)

        imgui.separator()
        _, state.show_axes = imgui.checkbox("Show Axes", state.show_axes)

        if imgui.button("Reset Transform", width=-1):
            state.reset_transform()

        imgui.separator()
        imgui.text("Translate")
        _, state.translate[0] = imgui.slider_float("X##trans", state.translate[0], -10, 10)
        _, state.translate[1] = imgui.slider_float("Y##trans", state.translate[1], -10, 10)
        _, state.translate[2] = imgui.slider_float("Z##trans", state.translate[2], -10, 10)

        imgui.separator()
        imgui.text("Scale")
        _, state.scale[0] = imgui.slider_float("X##scale", state.scale[0], 0.1, 10)
        _, state.scale[1] = imgui.slider_float("Y##scale", state.scale[1], 0.1, 10)
        _, state.scale[2] = imgui.slider_float("Z##scale", state.scale[2], 0.1, 10)

        imgui.separator()
        imgui.text("Color (RGB)")

        _, state.color[0] = imgui.slider_float("R", state.color[0], 0.0, 1.0)
        _, state.color[1] = imgui.slider_float("G", state.color[1], 0.0, 1.0)
        _, state.color[2] = imgui.slider_float("B", state.color[2], 0.0, 1.0)

        imgui.color_button("Preview", *state.color, flags=0)

        imgui.end()

    def _update_scene_from_state(self):
        current_shape = self.state.shapes_2d[self.state.shape_idx]
        shape_changed = (self._managed_drawable is None or
                         self._managed_drawable.__class__.__name__ != current_shape)
        if shape_changed:
            try:
                shape_class = globals()[current_shape]
                self._managed_drawable = shape_class(
                    render_mode=self.state.render_modes[self.state.render_mode_idx],
                    vert_shader="(ignored)",
                    frag_shader="(ignored)"
                ).setup()
                self._managed_drawable.shape_name = current_shape
                self.drawables = [self._managed_drawable]
            except Exception as e:
                print(f"Error creating shape: {e}")

        s = self.state
        transform = translate(s.translate) @ scale(s.scale)
        if self._managed_drawable:
            self._managed_drawable.transform = transform
        if hasattr(self._managed_drawable, "color"):
            self._managed_drawable.color = self.state.color
        elif hasattr(self._managed_drawable, "set_color"):
            self._managed_drawable.set_color(self.state.color)


    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            self.render_ui(self.state)
            self._update_scene_from_state()

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            if self.state.show_axes:
                self.axes.draw(projection, view, np.eye(4))

            for drawable in self.drawables:
                drawable.draw(projection, view, drawable.transform)

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)

        self.impl.shutdown()
        imgui.destroy_context()

    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.win, True)

    def on_mouse_move(self, win, xpos, ypos):
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])


def main():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    viewer = Viewer()
    viewer.run()
    glfw.terminate()


if __name__ == "__main__":
    main()
