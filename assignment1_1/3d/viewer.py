import OpenGL.GL as GL
import glfw
import numpy as np
from itertools import cycle
from tostudents.libs.transform import Trackball, translate, rotate, scale
import imgui
from imgui.integrations.glfw import GlfwRenderer

#from tostudents.shape2d.triangle2d import Triangle2D
from tostudents.shape2d.shape2d import *
from tostudents.shape3d.basic3d import *
from tostudents.main.axes import Axes


class UIState:
    def __init__(self):
        self.render_modes = ["Flat","Texture","Gouraud", "Phong", "Wireframe"]
        self.render_mode_idx = 0

        

        self.shapes_3d = [
            "Cylinder", "Cube", "Sphere", "Cone", 
            "Tetrahedron", "Torus", "Prism", "Equation"
        ]

        self.shape_idx = 0

        self.equation_str = "sin(x) * cos(y)"
        self.shape_idx = 0

        self.translate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.rotate = [0.0, 0.0, 0.0]

        
        # Thêm tham số cho Cylinder/Cone
        self.cylinder_r_bottom = 0.3
        self.cylinder_r_top = 0.3
        self.cylinder_segments = 32
        
        # Thêm tham số cho Torus
        self.torus_major_radius = 0.4
        self.torus_minor_radius = 0.15
        self.torus_major_segments = 32
        self.torus_minor_segments = 16
        
        # Thêm tham số cho Prism
        self.prism_sides = 6
        self.prism_height = 0.8
        self.prism_radius = 0.4
        
        # Thêm toggle cho axes
        self.show_axes = True

        self.color = [1.0, 1.0, 1.0]

    def reset_transform(self):
        self.translate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.rotate = [0.0, 0.0, 0.0]
        self.cylinder_r_bottom = 0.3
        self.cylinder_r_top = 0.3
        self.cylinder_segments = 32
        self.torus_major_radius = 0.4
        self.torus_minor_radius = 0.15
        self.torus_major_segments = 32
        self.torus_minor_segments = 16
        self.prism_sides = 6
        self.prism_height = 0.8
        self.prism_radius = 0.4

        self.color = [1.0, 1.0, 1.0]


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
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, 'Viewer with UI', None, None)
        glfw.make_context_current(self.win)

        # --- Initialize ImGui ---
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)
        self.make_colors()

        # --- Trackball setup ---
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # --- State ---
        self.state = UIState()
        self.drawables = []
        self._managed_drawable = None
        self._last_cylinder_params = None
        self._last_torus_params = None
        self._last_prism_params = None
        self._last_equation_str = None
        
        # --- Initialize Axes ---
        self.axes = Axes(
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag",
            length=2.0
        ).setup()

        # --- Setup callbacks ---
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

    def make_colors(self):
        style = imgui.get_style()
        style.window_rounding = 4.0
        colors = style.colors
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.12, 0.16, 0.20, 1.0)
        colors[imgui.COLOR_BUTTON] = (0.2, 0.48, 0.54, 1.0)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.3, 0.6, 0.7, 1.0)

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

            # Draw axes first (if enabled)
            if self.state.show_axes:
                self.axes.draw(projection, view, np.eye(4))

            # Draw main shape
            for drawable in self.drawables:
                drawable.draw(projection, view, drawable.transform)

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)

        self.impl.shutdown()
        imgui.destroy_context()

    def render_ui(self, state):
        sidebar_width = 320
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(sidebar_width, 700)
        imgui.begin("Control Panel", True)

        imgui.text("Render Mode:")
        changed, state.render_mode_idx = imgui.combo("##render_mode", state.render_mode_idx, state.render_modes)

        if state.render_modes[state.render_mode_idx] == "Flat":
            imgui.separator()
            imgui.text("Flat Color (RGB):")
            _, state.color[0] = imgui.slider_float("R", state.color[0], 0.0, 1.0)
            _, state.color[1] = imgui.slider_float("G", state.color[1], 0.0, 1.0)
            _, state.color[2] = imgui.slider_float("B", state.color[2], 0.0, 1.0)
            imgui.color_button("Preview", *state.color, flags=0)

        imgui.text("3D Shape:")
        _, state.shape_idx = imgui.combo("##shape3d", state.shape_idx, state.shapes_3d)


        # --- Show Axes Toggle ---
        imgui.separator()
        _, state.show_axes = imgui.checkbox("Show Axes", state.show_axes)

        # --- Cylinder/Cone parameters ---
        current_shapes = state.shapes_3d
        if current_shapes[state.shape_idx] == "Cylinder":
            imgui.separator()
            imgui.text("Cylinder Parameters:")
            
            _, state.cylinder_r_bottom = imgui.slider_float(
                "Bottom Radius", state.cylinder_r_bottom, 0.0, 1.0
            )
            _, state.cylinder_r_top = imgui.slider_float(
                "Top Radius", state.cylinder_r_top, 0.0, 1.0
            )
            _, state.cylinder_segments = imgui.slider_int(
                "Segments", state.cylinder_segments, 8, 64
            )
            
            imgui.text("Presets:")
            if imgui.button("Cylinder", width=95):
                state.cylinder_r_bottom = 0.3
                state.cylinder_r_top = 0.3
            imgui.same_line()
            if imgui.button("Cone", width=95):
                state.cylinder_r_bottom = 0.3
                state.cylinder_r_top = 0.0
            imgui.same_line()
            if imgui.button("Inv Cone", width=95):
                state.cylinder_r_bottom = 0.0
                state.cylinder_r_top = 0.3
        
        # --- Torus parameters ---
        if current_shapes[state.shape_idx] == "Torus":
            imgui.separator()
            imgui.text("Torus Parameters:")
            
            _, state.torus_major_radius = imgui.slider_float(
                "Major Radius (R)", state.torus_major_radius, 0.2, 0.8
            )
            _, state.torus_minor_radius = imgui.slider_float(
                "Minor Radius (r)", state.torus_minor_radius, 0.05, 0.4
            )
            _, state.torus_major_segments = imgui.slider_int(
                "Major Segments", state.torus_major_segments, 8, 64
            )
            _, state.torus_minor_segments = imgui.slider_int(
                "Minor Segments", state.torus_minor_segments, 8, 32
            )
            
            imgui.text("Presets:")
            if imgui.button("Donut", width=95):
                state.torus_major_radius = 0.4
                state.torus_minor_radius = 0.15
            imgui.same_line()
            if imgui.button("Thick Ring", width=95):
                state.torus_major_radius = 0.5
                state.torus_minor_radius = 0.25
            imgui.same_line()
            if imgui.button("Thin Ring", width=95):
                state.torus_major_radius = 0.6
                state.torus_minor_radius = 0.08
        
        # --- Prism parameters ---
        if current_shapes[state.shape_idx] == "Prism":
            imgui.separator()
            imgui.text("Prism Parameters:")
            
            _, state.prism_sides = imgui.slider_int(
                "Number of Sides", state.prism_sides, 3, 12
            )
            _, state.prism_height = imgui.slider_float(
                "Height", state.prism_height, 0.2, 2.0
            )
            _, state.prism_radius = imgui.slider_float(
                "Radius", state.prism_radius, 0.1, 0.8
            )
            
            imgui.text("Presets:")
            if imgui.button("Triangle", width=95):
                state.prism_sides = 3
            imgui.same_line()
            if imgui.button("Square", width=95):
                state.prism_sides = 4
            imgui.same_line()
            if imgui.button("Hexagon", width=95):
                state.prism_sides = 6
                
        # --- Equation parameters ---
        if current_shapes[state.shape_idx] == "Equation":
            imgui.separator()
            imgui.text("Equation z = f(x, y):")
            changed, state.equation_str = imgui.input_text(
                "##equation", 
                state.equation_str, 
                256
            )
            
            imgui.text("Examples:")
            if imgui.button("sin(x)*cos(y)", width=-1):
                state.equation_str = "sin(x) * cos(y)"
            if imgui.button("x**2 + y**2", width=-1):
                state.equation_str = "x**2 + y**2"
            if imgui.button("sin(sqrt(x**2+y**2))", width=-1):
                state.equation_str = "sin(sqrt(x**2 + y**2))"

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
        imgui.text("Rotate (degrees)")
        _, state.rotate[0] = imgui.slider_float("X##rot", state.rotate[0], -180, 180)
        _, state.rotate[1] = imgui.slider_float("Y##rot", state.rotate[1], -180, 180)
        _, state.rotate[2] = imgui.slider_float("Z##rot", state.rotate[2], -180, 180)


        imgui.end()

    def _update_scene_from_state(self):
        """Sync shape with UI"""
        
        current_shape = self.state.shapes_3d[self.state.shape_idx]

        
        current_cylinder_params = (
            self.state.cylinder_r_bottom,
            self.state.cylinder_r_top,
            self.state.cylinder_segments
        )
        cylinder_params_changed = (self._last_cylinder_params != current_cylinder_params)
        
        current_torus_params = (
            self.state.torus_major_radius,
            self.state.torus_minor_radius,
            self.state.torus_major_segments,
            self.state.torus_minor_segments
        )
        torus_params_changed = (self._last_torus_params != current_torus_params)
        
        current_prism_params = (
            self.state.prism_sides,
            self.state.prism_height,
            self.state.prism_radius
        )
        prism_params_changed = (self._last_prism_params != current_prism_params)
        
        equation_changed = (self._last_equation_str != self.state.equation_str)
        
        shape_changed = (self._managed_drawable is None or 
                        self._managed_drawable.__class__.__name__ != current_shape)
        
        needs_recreate = (
            shape_changed or 
            (current_shape == "Cylinder" and cylinder_params_changed) or
            (current_shape == "Torus" and torus_params_changed) or
            (current_shape == "Prism" and prism_params_changed) or
            (current_shape == "Equation" and equation_changed)
        )
        
        if needs_recreate:
            try:
                if current_shape in [
                    "Triangle2D", "Rectangle2D", "Pentagon2D", "Hexagon2D",
                    "Circle2D", "Ellipse2D", "Trapezoid2D", "Star2D", "Arrow2D"
                ]:
                    shape_class = globals()[current_shape]
                    self._managed_drawable = shape_class(
                        render_mode=self.state.render_modes[self.state.render_mode_idx],
                        vert_shader="(ignored)",
                        frag_shader="(ignored)"
                    ).setup()

                elif current_shape == "Cylinder":
                    self._managed_drawable = Cylinder2(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag",
                        n=self.state.cylinder_segments,
                        r_bottom=self.state.cylinder_r_bottom,
                        r_top=self.state.cylinder_r_top
                    ).setup()
                    self._last_cylinder_params = current_cylinder_params
                    
                elif current_shape == "Cube":
                    self._managed_drawable = Cube(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag"
                    ).setup()

                
                    
                elif current_shape == "Sphere":
                    self._managed_drawable = Sphere(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag"
                    ).setup()
                    
                elif current_shape == "Cone":
                    self._managed_drawable = Cone(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag"
                    ).setup()
                    
                elif current_shape == "Tetrahedron":
                    self._managed_drawable = Tetrahedron(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag",
                        size=0.5
                    ).setup()
                
                elif current_shape == "Torus":
                    self._managed_drawable = Torus(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag",
                        major_segments=self.state.torus_major_segments,
                        minor_segments=self.state.torus_minor_segments,
                        major_radius=self.state.torus_major_radius,
                        minor_radius=self.state.torus_minor_radius
                    ).setup()
                    self._last_torus_params = current_torus_params
                
                elif current_shape == "Prism":
                    self._managed_drawable = Prism(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag",
                        n_sides=self.state.prism_sides,
                        height=self.state.prism_height,
                        radius=self.state.prism_radius
                    ).setup()
                    self._last_prism_params = current_prism_params
                    
                elif current_shape == "Equation":
                    from tostudents.assignment1_1 import EquationMesh
                    self._managed_drawable = EquationMesh(
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.vert",
                        "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/3d/shaders/gouraud.frag",
                        func_str=self.state.equation_str,
                        n=50
                    ).setup()
                    self._last_equation_str = self.state.equation_str
                else:
                    raise ValueError(f"Unknown shape: {current_shape}")

                self._managed_drawable.shape_name = current_shape
                self.drawables = [self._managed_drawable]
                
            except Exception as e:
                print(f"Error creating shape: {e}")
                pass
            
        s = self.state
       
        transform = (
    translate(s.translate)
    @ rotate(axis=[1, 0, 0], angle=s.rotate[0])
    @ rotate(axis=[0, 1, 0], angle=s.rotate[1])
    @ rotate(axis=[0, 0, 1], angle=s.rotate[2])
    @ scale(s.scale)
)

        if self._managed_drawable:
            self._managed_drawable.transform = transform
        # --- Apply flat color if in Flat mode ---
        if s.render_modes[s.render_mode_idx] == "Flat":
            if hasattr(self._managed_drawable, "color"):
                self._managed_drawable.color = s.color
            elif hasattr(self._managed_drawable, "set_color"):
                self._managed_drawable.set_color(s.color)


    # --- Event handling ---
    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_A:
                self.state.show_axes = not self.state.show_axes

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