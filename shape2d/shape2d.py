import numpy as np
import OpenGL.GL as GL
from math import sin, cos, pi
from tostudents.libs.shader import *
from tostudents.libs.buffer import *


# ============================================================
# 🔹 BASE CLASS CHUNG CHO TẤT CẢ CÁC HÌNH 2D
# ============================================================

class Shape2DBase:
    def __init__(self, vertices, indices, colors, render_mode="Flat"):
        self.render_mode = render_mode
        self.vertices = vertices.astype(np.float32)
        self.indices = indices.astype(np.int32)
        self.colors = colors.astype(np.float32)
        self.vao = VAO()

        # --- Shader chọn theo chế độ ---
        if render_mode == "Flat":
            vs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/flat2d.vert"
            fs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/flat2d.frag"
        elif render_mode == "Gouraud":
            vs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/gouraud2d.vert"
            fs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/gouraud2d.frag"
        elif render_mode == "Phong":
            vs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/phong2d.vert"
            fs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/phong2d.frag"
        elif render_mode == "Texture":
            vs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/texture2d.vert"
            fs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/texture2d.frag"
        else:
            vs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/flat2d.vert"
            fs = "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/shader2d/flat2d.frag"


        self.shader = Shader(vs, fs)
        self.uma = UManager(self.shader)
        self.flat_color = np.array([0.9, 0.4, 0.2], dtype=np.float32)  # Màu mặc định


    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3)
        self.vao.add_vbo(1, self.colors, ncomponents=3)
        self.vao.add_ebo(self.indices)
        return self
    def set_color(self, rgb):
        self.flat_color = np.array(rgb, dtype=np.float32)


    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(view, 'view', True)
        self.uma.upload_uniform_matrix4fv(model, 'model', True)

        # --- Wireframe Mode ---
        if self.render_mode == "Wireframe":
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glLineWidth(1.0)

            # Dùng màu trắng rõ ràng để thấy viền
            loc = GL.glGetUniformLocation(self.shader.render_idx, "uColor")
            if loc != -1:
                GL.glUniform3f(loc, 1.0, 1.0, 1.0)
        else:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            if self.render_mode == "Flat":
                loc = GL.glGetUniformLocation(self.shader.render_idx, "uColor")
                if loc != -1:
                    GL.glUniform3fv(loc, 1, self.flat_color)

        # --- Draw call ---
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

        # Reset lại polygon mode để không ảnh hưởng frame sau
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)



# ============================================================
# 🔺 TRIANGLE
# ============================================================
class Triangle2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat"):
        vertices = np.array([[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]])
        indices = np.array([0, 1, 2])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        super().__init__(vertices, indices, colors, render_mode)


# ============================================================
# ⬜ RECTANGLE
# ============================================================
class Rectangle2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat"):
        vertices = np.array([[-0.8, -0.5, 0], [0.8, -0.5, 0], [0.8, 0.5, 0], [-0.8, 0.5, 0]])
        indices = np.array([0, 1, 2, 2, 3, 0])
        colors = np.random.rand(4, 3)
        super().__init__(vertices, indices, colors, render_mode)


# ============================================================
# 🔷 PENTAGON
# ============================================================
class Pentagon2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat"):
        n, r = 5, 0.5
        vertices = np.array([[r*cos(2*pi*i/n), r*sin(2*pi*i/n), 0] for i in range(n)])
        indices = np.concatenate([[0, i, i+1] for i in range(1, n-1)])
        colors = np.random.rand(n, 3)
        super().__init__(vertices, indices, colors, render_mode)


# ============================================================
# 🛑 HEXAGON
# ============================================================
class Hexagon2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat"):
        n, r = 6, 0.5
        vertices = np.array([[r*cos(2*pi*i/n), r*sin(2*pi*i/n), 0] for i in range(n)])
        indices = np.concatenate([[0, i, i+1] for i in range(1, n-1)])
        colors = np.random.rand(n, 3)
        super().__init__(vertices, indices, colors, render_mode)


# ============================================================
# ⚪ CIRCLE (FIXED)
# ============================================================
class Circle2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat", radius=0.5):
        n = 120
        r = radius
        vertices = np.array([[0, 0, 0]] + [[r*cos(2*pi*i/n), r*sin(2*pi*i/n), 0] for i in range(n)])
        indices = np.concatenate([[0, i, i+1] for i in range(1, n)] + [[0, n, 1]])
        colors = np.random.rand(n+1, 3)
        super().__init__(vertices, indices, colors, render_mode)



# ============================================================
# 🟣 ELLIPSE (FIXED)
# ============================================================
class Ellipse2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat", a=0.6, b=0.4):
        n = 60
        vertices = np.array([[0, 0, 0]] + [[a*cos(2*pi*i/n), b*sin(2*pi*i/n), 0] for i in range(n)])
        indices = np.concatenate([[0, i, i+1] for i in range(1, n)] + [[0, n, 1]])
        colors = np.random.rand(n+1, 3)
        super().__init__(vertices, indices, colors, render_mode)



# ============================================================
# ⏹ TRAPEZOID
# ============================================================
class Trapezoid2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat"):
        vertices = np.array([
            [-0.6, -0.4, 0], [0.6, -0.4, 0], [0.3, 0.4, 0], [-0.3, 0.4, 0]
        ])
        indices = np.array([0, 1, 2, 2, 3, 0])
        colors = np.random.rand(4, 3)
        super().__init__(vertices, indices, colors, render_mode)


# ============================================================
# ============================================================
# ⭐ STAR (FIXED - Proper center-based triangulation)
# ============================================================
class Star2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat", n=5, R=0.5, r=0.2):
        vertices = [[0, 0, 0]]
        for i in range(2*n):
            ang = i*pi/n - pi/2
            radius = R if i % 2 == 0 else r
            vertices.append([radius*cos(ang), radius*sin(ang), 0])
        vertices = np.array(vertices)
        indices = np.concatenate([[0, i, i+1] for i in range(1, 2*n)] + [[0, 2*n, 1]])
        colors = np.random.rand(len(vertices), 3)
        super().__init__(vertices, indices, colors, render_mode)



# ============================================================
# 🏹 ARROW
# ============================================================
class Arrow2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat"):
        vertices = np.array([
            [-0.5, -0.1, 0], [0.3, -0.1, 0], [0.3, -0.3, 0],
            [0.7, 0.0, 0], [0.3, 0.3, 0], [0.3, 0.1, 0], [-0.5, 0.1, 0]
        ])
        indices = np.array([0,1,5, 0,5,6, 1,2,4, 4,5,1, 2,3,4])
        colors = np.random.rand(len(vertices), 3)
        super().__init__(vertices, indices, colors, render_mode)

class RegularPolygon2D(Shape2DBase):
    def __init__(self, vert_shader, frag_shader, render_mode="Flat", n=6, r=0.5):
        vertices = np.array([[r * np.cos(2*np.pi*i/n), r * np.sin(2*np.pi*i/n), 0] for i in range(n)])
        indices = np.concatenate([[0, i, (i+1) % n] for i in range(1, n-1)])
        colors = np.random.rand(n, 3)
        super().__init__(vertices, indices, colors, render_mode)
