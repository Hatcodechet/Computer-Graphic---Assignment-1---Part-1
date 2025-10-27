import numpy as np
import numexpr as ne
from sympy import symbols, sympify
from OpenGL import GL
from tostudents.libs.shader import *
from tostudents.libs import transform as T
from tostudents.libs.buffer import *
class EquationSurface(object):
    def __init__(self, vert_shader, frag_shader, func_str="sin(x)*cos(y)",
             x_range=(-5,5), y_range=(-5,5), n=80):
        self.x_range = x_range
        self.y_range = y_range
        xs = np.linspace(x_range[0], x_range[1], n)
        ys = np.linspace(y_range[0], y_range[1], n)
        grid = np.array([[x, y, 0] for y in ys for x in xs], dtype=np.float32)

        # --- 2. Tính z = f(x, y)
        x, y = symbols("x y")
        expr = sympify(func_str)
        expr_str = str(expr)
        z_values = ne.evaluate(expr_str, local_dict={'x': grid[:, 0], 'y': grid[:, 1]})
        grid[:, 2] = z_values
        self.vertices = grid
        z_scale = 0.8  # hoặc 5.0 nếu bạn muốn mặt lồi lên rõ hơn
        self.vertices[:, 2] *= z_scale

        # --- 3. Tạo indices
        indices = []
        for j in range(n - 1):
            for i in range(n - 1):
                idx = j * n + i
                indices += [idx, idx + 1, idx + n, idx + 1, idx + n + 1, idx + n]
        self.indices = np.array(indices, dtype=np.uint32)

        # --- 4. Màu theo độ cao
        zmin, zmax = np.min(z_values), np.max(z_values)
        normalized = (z_values - zmin) / (zmax - zmin + 1e-8)
        self.colors = np.array([[val, 1 - val, 0.4 + 0.3 * val] for val in normalized], dtype=np.float32)

        # --- 5. Normal (pháp tuyến) — gần đúng
        self.normals = np.zeros_like(self.vertices, dtype=np.float32)
        for j in range(1, n - 1):
            for i in range(1, n - 1):
                p = self.vertices[j * n + i]
                px = self.vertices[j * n + (i + 1)] - self.vertices[j * n + (i - 1)]
                py = self.vertices[(j + 1) * n + i] - self.vertices[(j - 1) * n + i]
                nrm = np.cross(px, py)
                self.normals[j * n + i] = nrm / (np.linalg.norm(nrm) + 1e-8)

        # --- 6. Shader + VAO + Uniform manager
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()
        self.transform = np.eye(4)

    def setup(self):
        """Chuẩn bị dữ liệu GPU"""
        self.vao.add_vbo(0, self.vertices, ncomponents=3)
        self.vao.add_vbo(1, self.colors, ncomponents=3)
        self.vao.add_vbo(2, self.normals, ncomponents=3)
        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        """Vẽ bề mặt"""
        GL.glUseProgram(self.shader.render_idx)
        
        if model is None:
            model = np.eye(4, dtype=np.float32)

        self.uma.upload_uniform_matrix4fv(projection, "projection", True)
        self.uma.upload_uniform_matrix4fv(model, "model", True)
        self.uma.upload_uniform_matrix4fv(view, "view", True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def set_color(self, rgb):
        """Cho phép đổi màu từ Viewer Flat mode"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
