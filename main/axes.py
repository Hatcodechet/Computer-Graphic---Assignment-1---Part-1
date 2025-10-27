from tostudents.libs.shader import *
from tostudents.libs.buffer import *
import numpy as np
from OpenGL import GL

class Axes(object):
    def __init__(self, vert_shader, frag_shader, length=2.0):
        """
        Hệ trục tọa độ XYZ
        length: độ dài mỗi trục
        """
        self.length = length
        
        # 6 đỉnh: gốc và 3 đầu trục
        self.vertices = np.array([
            # Trục X (đỏ)
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            
            # Trục Y (xanh lá)
            [0.0, 0.0, 0.0],
            [0.0, length, 0.0],
            
            # Trục Z (xanh dương)
            [0.0, 0.0, 0.0],
            [0.0, 0.0, length],
        ], dtype=np.float32)
        
        # Màu cho mỗi đỉnh
        self.colors = np.array([
            # Trục X - Đỏ
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            
            # Trục Y - Xanh lá
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            
            # Trục Z - Xanh dương
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Indices cho 3 đường thẳng
        self.indices = np.array([
            0, 1,  # Trục X
            2, 3,  # Trục Y
            4, 5,  # Trục Z
        ], dtype=np.uint32)
        
        # Shader & VAO
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.transform = np.eye(4)

    def setup(self):
        """Setup VAO"""
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        """Vẽ hệ trục"""
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            model = np.eye(4, dtype=np.float32)

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(model, 'model', True)
        self.uma.upload_uniform_matrix4fv(view, 'view', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        
        # Vẽ các đường thẳng (không dùng glLineWidth vì không support trên macOS Core Profile)
        GL.glDrawElements(GL.GL_LINES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)