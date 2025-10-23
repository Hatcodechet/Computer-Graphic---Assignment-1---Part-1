import OpenGL.GL as GL
import glfw
import numpy as np
from tostudents.libs.transform import Trackball, translate, rotate, scale
import imgui
from imgui.integrations.glfw import GlfwRenderer
from tostudents.shape3d.basic3d import Sphere
from tostudents.main.axes import Axes
import os


class UIState:
    def __init__(self):
        # --- Cấu hình mặc định ---
        self.show_axes = True

        # --- Transform ---
        self.translate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.rotate = [0.0, 0.0, 0.0]

        # --- Texture ---
        self.texture_path = (
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/"
            "Gradient-Descent-Visualizer/resources/textures/earth.jpg"
        )

    def reset_transform(self):
        self.translate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.rotate = [0.0, 0.0, 0.0]


class Viewer:
    def __init__(self, width=1200, height=900):
        self.width = width
        self.height = height

        # --- GLFW setup ---
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, "Sphere Texture Viewer", None, None)
        if not self.win:
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.win)

        # --- ImGui setup ---
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)

        # --- Trackball camera ---
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # --- State ---
        self.state = UIState()

        # --- Axes setup ---
        shader_dir = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/"
        self.axes = Axes(
            shader_dir + "gouraud.vert",
            shader_dir + "gouraud.frag",
            length=2.0
        ).setup()

        # --- Sphere setup ---
        vert = shader_dir + "texture.vert"
        frag = shader_dir + "texture.frag"
        self.sphere = Sphere(vert, frag, texture_path=self.state.texture_path).setup()

        # --- OpenGL state ---
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.05, 0.05, 0.08, 1.0)

        # --- Callbacks ---
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

    # ================================================================
    # -------------------------- MAIN LOOP ---------------------------
    # ================================================================
    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            # --- UI ---
            self.render_ui()

            # --- Render scene ---
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            # --- Draw axes ---
            if self.state.show_axes:
                self.axes.draw(projection, view, np.eye(4))

            # --- Transform ---
            s = self.state
            transform = (
                translate(s.translate)
                @ rotate(axis=[1, 0, 0], angle=s.rotate[0])
                @ rotate(axis=[0, 1, 0], angle=s.rotate[1])
                @ rotate(axis=[0, 0, 1], angle=s.rotate[2])
                @ scale(s.scale)
            )

            # --- Draw sphere ---
            self.sphere.draw(projection, view, transform)

            # --- Render ImGui ---
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)

        # Cleanup
        self.impl.shutdown()
        imgui.destroy_context()
        glfw.terminate()

    # ================================================================
    # -------------------------- UI PANEL ----------------------------
    # ================================================================
    def render_ui(self):
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(320, 400)
        imgui.begin("Sphere Texture Controls", True)

        imgui.text("Texture Path:")
        imgui.text_wrapped(self.state.texture_path)

        if imgui.button("Load New Texture", width=-1):
            try:
                import easygui
                path = easygui.fileopenbox(title="Select Texture", default="*.jpg")
                if path:
                    self.state.texture_path = os.path.abspath(path)
                    self.sphere.load_texture(self.state.texture_path)
                    print(f"[INFO] Loaded texture: {self.state.texture_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load texture: {e}")

        imgui.separator()
        _, self.state.show_axes = imgui.checkbox("Show Axes", self.state.show_axes)

        imgui.separator()
        imgui.text("Transform:")
        _, self.state.translate[0] = imgui.slider_float("Trans X", self.state.translate[0], -5, 5)
        _, self.state.translate[1] = imgui.slider_float("Trans Y", self.state.translate[1], -5, 5)
        _, self.state.translate[2] = imgui.slider_float("Trans Z", self.state.translate[2], -5, 5)

        _, self.state.scale[0] = imgui.slider_float("Scale X", self.state.scale[0], 0.1, 3.0)
        _, self.state.scale[1] = imgui.slider_float("Scale Y", self.state.scale[1], 0.1, 3.0)
        _, self.state.scale[2] = imgui.slider_float("Scale Z", self.state.scale[2], 0.1, 3.0)

        _, self.state.rotate[0] = imgui.slider_float("Rotate X", self.state.rotate[0], -180, 180)
        _, self.state.rotate[1] = imgui.slider_float("Rotate Y", self.state.rotate[1], -180, 180)
        _, self.state.rotate[2] = imgui.slider_float("Rotate Z", self.state.rotate[2], -180, 180)

        if imgui.button("Reset Transform", width=-1):
            self.state.reset_transform()

        imgui.end()

    # ================================================================
    # ------------------------- INTERACTION --------------------------
    # ================================================================
    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS:
            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(self.win, True)
            elif key == glfw.KEY_A:
                self.state.show_axes = not self.state.show_axes

    def on_mouse_move(self, win, xpos, ypos):
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        elif glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])


def main():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    viewer = Viewer()
    viewer.run()


if __name__ == "__main__":
    main()
