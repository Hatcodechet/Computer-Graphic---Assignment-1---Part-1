# viewer_equation.py
import glfw
import OpenGL.GL as GL
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numexpr as ne
import matplotlib.pyplot as plt

from tostudents.libs.transform import Trackball
from tostudents.assignment1_1.shape3d.mesh import EquationSurface


class UIState:
    def __init__(self):
        self.render_modes = ["Solid", "Wireframe"]
        self.render_mode_idx = 0

        self.view_modes = ["3D View", "2D Contour"]
        self.view_mode_idx = 0

        self.func_str = "sin(x)*cos(y)"
        self.x_range = [-5.0, 5.0]
        self.y_range = [-5.0, 5.0]
        self.n = 60
        self.critical_points = []


class ViewerEquation:
    def __init__(self, width=1200, height=900):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.win = glfw.create_window(width, height, "Equation Viewer", None, None)
        glfw.make_context_current(self.win)

        imgui.create_context()
        self.impl = GlfwRenderer(self.win)

        self.trackball = Trackball()
        self.mouse = (0, 0)

        self.state = UIState()
        


        shader_dir = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/assignment1_1/shape3d/shaders/"
        vert = shader_dir + "wf.vert"
        frag = shader_dir + "wf.frag"
        self.surface = EquationSurface(
            vert,
            frag,
            func_str=self.state.func_str,
            n=self.state.n,
            x_range=tuple(self.state.x_range),
            y_range=tuple(self.state.y_range),
        ).setup()
        self._shader_vert = vert
        self._shader_frag = frag

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.05, 0.05, 0.08, 1.0)

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # Matplotlib figure cho chế độ 2D
        #import matplotlib.pyplot as plt
        # Matplotlib figure cho chế độ 2D (đặt SAU phần khởi tạo OpenGL)
        plt.ion()
        self._fig = plt.figure("2D Contour", figsize=(6, 5), dpi=100)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._cbar = None
        
        # **THÊM PAN + ZOOM**
        self._pan_start = None
        self._fig.canvas.mpl_connect('scroll_event', self.on_mpl_scroll)
        self._fig.canvas.mpl_connect('button_press_event', self.on_mpl_press)
        self._fig.canvas.mpl_connect('button_release_event', self.on_mpl_release)
        self._fig.canvas.mpl_connect('motion_notify_event', self.on_mpl_motion)
        
        plt.show(block=False)
    def on_mpl_scroll(self, event):
        if event.inaxes != self._ax:
            return
        
        zoom_factor = 1.2 if event.button == 'up' else 0.8
        
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        
        new_xlim = [xdata - (xdata - xlim[0]) * zoom_factor,
                    xdata + (xlim[1] - xdata) * zoom_factor]
        new_ylim = [ydata - (ydata - ylim[0]) * zoom_factor,
                    ydata + (ylim[1] - ydata) * zoom_factor]
        
        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        self._fig.canvas.draw_idle()

    def on_mpl_press(self, event):
        """Bắt đầu pan khi nhấn chuột phải"""
        if event.button == 3 and event.inaxes == self._ax:  # Right click
            self._pan_start = (event.xdata, event.ydata)

    def on_mpl_release(self, event):
        """Kết thúc pan"""
        self._pan_start = None

    def on_mpl_motion(self, event):
        """Kéo thả để pan"""
        if self._pan_start is None or event.inaxes != self._ax:
            return
        
        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata
        
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        
        self._ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self._ax.set_ylim([ylim[0] + dy, ylim[1] + dy])
        
        self._fig.canvas.draw_idle()

    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            self.render_ui()

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            if self.state.view_modes[self.state.view_mode_idx] == "3D View":
                # Polygon mode
                if self.state.render_modes[self.state.render_mode_idx] == "Wireframe":
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                else:
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

                # Vẽ surface
                self.surface.draw(projection, view, np.eye(4))

                # Setup shader và uniforms cho việc vẽ điểm
                GL.glUseProgram(self.surface.shader.render_idx)
                proj_loc = GL.glGetUniformLocation(self.surface.shader.render_idx, "projection")
                mv_loc = GL.glGetUniformLocation(self.surface.shader.render_idx, "modelview")
                if proj_loc != -1:
                    GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_TRUE, projection)
                if mv_loc != -1:
                    GL.glUniformMatrix4fv(mv_loc, 1, GL.GL_TRUE, view)

                # **VẼ TẤT CẢ CÁC ĐIỂM CRITICAL**
                if self.state.critical_points:
                    for pt in self.state.critical_points:
                        x, y, color_hex, label = pt
                        
                        # Tính z = f(x, y)
                        try:
                            z = float(ne.evaluate(self.state.func_str, local_dict={'x': x, 'y': y}))
                        except Exception:
                            z = 0.0

                        # Convert hex color to RGB (0-1 range)
                        color_hex = color_hex.lstrip('#')
                        r = int(color_hex[0:2], 16) / 255.0
                        g = int(color_hex[2:4], 16) / 255.0
                        b = int(color_hex[4:6], 16) / 255.0

                        vertex = np.array([[x, y, z]], dtype=np.float32)
                        color = np.array([[r, g, b]], dtype=np.float32)

                        # Tạo VAO/VBO cho điểm này
                        vao = GL.glGenVertexArrays(1)
                        vbo_pos = GL.glGenBuffers(1)
                        vbo_col = GL.glGenBuffers(1)

                        GL.glBindVertexArray(vao)

                        # attrib 0: position
                        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
                        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex.nbytes, vertex, GL.GL_STATIC_DRAW)
                        GL.glEnableVertexAttribArray(0)
                        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

                        # attrib 1: color
                        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_col)
                        GL.glBufferData(GL.GL_ARRAY_BUFFER, color.nbytes, color, GL.GL_STATIC_DRAW)
                        GL.glEnableVertexAttribArray(1)
                        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

                        GL.glPointSize(12)
                        GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                        GL.glBindVertexArray(0)
                        GL.glPointSize(1)

                        GL.glDeleteBuffers(2, [vbo_pos, vbo_col])
                        GL.glDeleteVertexArrays(1, [vao])
                else:
                    # Nếu không có critical points, vẽ điểm tâm (0,0) như mặc định
                    try:
                        z0 = float(ne.evaluate(self.state.func_str, local_dict={'x': 0.0, 'y': 0.0}))
                    except Exception:
                        z0 = 0.0

                    center_vertex = np.array([[0.0, 0.0, z0]], dtype=np.float32)
                    center_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

                    vao = GL.glGenVertexArrays(1)
                    vbo_pos = GL.glGenBuffers(1)
                    vbo_col = GL.glGenBuffers(1)

                    GL.glBindVertexArray(vao)

                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, center_vertex.nbytes, center_vertex, GL.GL_STATIC_DRAW)
                    GL.glEnableVertexAttribArray(0)
                    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_col)
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, center_color.nbytes, center_color, GL.GL_STATIC_DRAW)
                    GL.glEnableVertexAttribArray(1)
                    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

                    GL.glPointSize(10)
                    GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                    GL.glBindVertexArray(0)
                    GL.glPointSize(1)

                    GL.glDeleteBuffers(2, [vbo_pos, vbo_col])
                    GL.glDeleteVertexArrays(1, [vao])

            else:
                # Chế độ 2D Contour
                self.show_contour()

            # Reset polygon mode
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)

        self.impl.shutdown()
        imgui.destroy_context()
        glfw.terminate()
        try:
            plt.close(self._fig)
        except Exception:
            pass

    def render_ui(self):
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(420, 700)
        imgui.begin("Equation Function Selector", True)

        imgui.text("View Mode:")
        _, self.state.view_mode_idx = imgui.combo("##view_mode", self.state.view_mode_idx, self.state.view_modes)

        imgui.text("Render Mode:")
        _, self.state.render_mode_idx = imgui.combo("##render_mode", self.state.render_mode_idx, self.state.render_modes)

        imgui.separator()
        imgui.text("Select Predefined Function:")

        presets = [
            {
                "name": "Himmelblau Function",
                "expr": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
                "x_range": [-6.0, 6.0],
                "y_range": [-6.0, 6.0],
                # **THÊM: 4 điểm cực tiểu**
                "critical_points": [
                    (3.0, 2.0, "#FF0000", "Min 1"),           # đỏ
                    (-2.805118, 3.131312, "#FF6B6B", "Min 2"), # đỏ nhạt
                    (-3.779310, -3.283186, "#9333EA", "Min 3"), # tím
                    (3.584428, -1.848126, "#FF8C00", "Min 4"),  # cam
                ]
            },
            {
                "name": "Rosenbrock Function",
                "expr": "100*(y - x**2)**2 + (1 - x)**2",
                "x_range": [-2.0, 2.0],
                "y_range": [-1.0, 3.0],
                "critical_points": [
                    (1.0, 1.0, "#00FF00", "Global Min")  # điểm (1,1) là cực tiểu
                ]
            },
            {
                "name": "Quadratic Bowl",
                "expr": "x**2 + y**2",
                "x_range": [-5.0, 5.0],
                "y_range": [-5.0, 5.0],
                "critical_points": [
                    (0.0, 0.0, "#0000FF", "Min")  # tâm là cực tiểu
                ]
            },
            {
                "name": "Booth Function",
                "expr": "(x + 2*y - 7)**2 + (2*x + y - 5)**2",
                "x_range": [-10.0, 10.0],
                "y_range": [-10.0, 10.0],
                "critical_points": [
                    (1.0, 3.0, "#00FFFF", "Global Min")  # điểm (1,3)
                ]
            },
        ]

        if not hasattr(self, "selected_func"):
            self.selected_func = 2  # default Quadratic

        for i, func in enumerate(presets):
            clicked = imgui.radio_button(func["name"], self.selected_func == i)
            if clicked:
                self.selected_func = i
                self.state.func_str = func["expr"]
                self.state.x_range = list(func["x_range"])
                self.state.y_range = list(func["y_range"])
                self.state.critical_points = func.get("critical_points", [])  # **THÊM**
                self.update_surface()

            imgui.indent(20)
            imgui.text(f"f(x,y) = {func['expr']}")
            imgui.text("X Range:")
            imgui.text(f"{func['x_range'][0]}  →  {func['x_range'][1]}")
            imgui.text("Y Range:")
            imgui.text(f"{func['y_range'][0]}  →  {func['y_range'][1]}")
            imgui.unindent(20)
            imgui.separator()

        imgui.text("Custom Function (optional):")
        _, self.state.func_str = imgui.input_text("##func", self.state.func_str, 256)

        _, self.state.x_range[0] = imgui.slider_float("X min", self.state.x_range[0], -10, 0)
        _, self.state.x_range[1] = imgui.slider_float("X max", self.state.x_range[1], 0, 10)
        _, self.state.y_range[0] = imgui.slider_float("Y min", self.state.y_range[0], -10, 0)
        _, self.state.y_range[1] = imgui.slider_float("Y max", self.state.y_range[1], 0, 10)

        _, self.state.n = imgui.slider_int("Resolution", self.state.n, 20, 200)

        if imgui.button("Update Surface", width=-1):
            self.update_surface()

        if imgui.button("Reset 2D View", width=-1):
            self.reset_contour_view()

        imgui.end()

    def update_surface(self):
        try:
            self.surface = EquationSurface(
                self._shader_vert,
                self._shader_frag,
                func_str=self.state.func_str,
                n=self.state.n,
                x_range=tuple(self.state.x_range),
                y_range=tuple(self.state.y_range),
            ).setup()
            print(f"[INFO] Updated surface: {self.state.func_str}")
        except Exception as e:
            print(f"[ERROR] Failed to update surface: {e}")

    def show_contour(self):
        func = self.state.func_str
        x_min, x_max = self.state.x_range
        y_min, y_max = self.state.y_range
        n = max(32, int(self.state.n))

        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(x, y)

        try:
            Z = ne.evaluate(func, local_dict={"x": X, "y": Y})
        except Exception as e:
            print(f"[ERROR] Invalid function: {e}")
            return

        Z = np.nan_to_num(Z, nan=np.nanmedian(Z))

        ax = self._ax
        ax.clear()
        
        ax.set_facecolor('white')
        self._fig.patch.set_facecolor('white')

        # Vẽ contour
        levels = 25
        cs = ax.contour(
            X, Y, Z,
            levels=levels,
            colors='#e0565b',
            linewidths=1.2,
            linestyles='solid'
        )

        # **VẼ TẤT CẢ CÁC ĐIỂM CRITICAL**
        for pt in self.state.critical_points:
            x, y, color, label = pt
            ax.plot([x], [y], "o", color=color, markersize=10, 
                    zorder=5, markeredgecolor='white', markeredgewidth=1.5)
            # Thêm label (tùy chọn)
            # ax.text(x, y + 0.3, label, ha='center', fontsize=8, color=color)

        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        
        ax.set_title(f"2D Contour of z = {func}", fontsize=10, pad=10)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        try:
            self._fig.canvas.manager.window.lift()
            self._fig.canvas.manager.window.attributes('-topmost', 1)
            self._fig.canvas.manager.window.attributes('-topmost', 0)
        except Exception:
            pass

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()





    # Events
    def on_key(self, win, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.win, True)

    def on_mouse_move(self, win, xpos, ypos):
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, dx, dy):
        self.trackball.zoom(dy, glfw.get_window_size(win)[1])


def main():
    viewer = ViewerEquation()
    viewer.run()


if __name__ == "__main__":
    main()
