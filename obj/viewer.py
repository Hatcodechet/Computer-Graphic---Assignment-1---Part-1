import OpenGL.GL as GL
import glfw
import numpy as np
from itertools import cycle
from libs.transform import Trackball
from Obj.load import ObjLoader


class Viewer:
    def __init__(self, width=800, height=800):

        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # Version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, 'Mercedes Viewer', None, None)

        # Make window's OpenGL context current
        glfw.make_context_current(self.win)

        # Initialize trackball for camera control
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # Register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # Print OpenGL information
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # Initialize GL settings
        GL.glClearColor(0.2, 0.2, 0.25, 1.0)  # Dark blue-gray background
        GL.glEnable(GL.GL_DEPTH_TEST)         # Enable depth testing for 3D
        GL.glDepthFunc(GL.GL_LESS)            # Depth test function

        # Initially empty list of objects to draw
        self.drawables = []

    def run(self):
        while not glfw.window_should_close(self.win):
            # Clear color and depth buffers
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Get current window size and calculate matrices
            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            # Draw all scene objects
            model = np.identity(4, dtype='float32')
            for drawable in self.drawables:
                drawable.draw(projection, view, model)

            # Swap front and back buffers (double buffering)
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            # Forward key events to drawables
            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

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
    viewer = Viewer()

    # Load Mercedes model with shaders
    model = ObjLoader("Obj/mercedes.obj", "Obj/mercedes.vert", "Obj/mercedes.frag")
    model.setup()
    viewer.add(model)

    # Start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()
