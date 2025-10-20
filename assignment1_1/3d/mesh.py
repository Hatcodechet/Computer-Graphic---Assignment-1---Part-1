from tostudents.libs.shader import *
from tostudents.libs.buffer import *
from OpenGL import GL
import numpy as np



'''
    objectSpace | worldSpace | viewSpace(camera) | clipSpace
Vertex            |
                |
                |
                |
                |








'''
class EquationMesh:
    def __init__(self, vert_shader, frag_shader, func_str="sin(x) * cos(y)", size=2.0, n=50):
        """
        func_str: chuỗi phương trình f(x, y)
        size: phạm vi [-size, size]
        n: số điểm theo mỗi chiều
        """
        self.func_str = func_str
        self.size = size
        self.n = n

        # Tạo lưới
        x = np.linspace(-size, size, n)
        y = np.linspace(-size, size, n)
        x, y = np.meshgrid(x, y)

        # Đánh giá phương trình với safe environment
        safe_env = {
            "np": np, 
            "x": x, 
            "y": y, 
            "sin": np.sin, 
            "cos": np.cos, 
            "exp": np.exp,
            "sqrt": np.sqrt,
            "tan": np.tan,
            "abs": np.abs,
            "log": np.log,
            "pi": np.pi
        }
        
        try:
            z = eval(func_str, safe_env)
            print(f"Equation evaluated successfully")
            print(f"Z range: [{z.min():.3f}, {z.max():.3f}]")
        except Exception as e:
            print(f"Lỗi phương trình: {e}")
            z = np.zeros_like(x)

        # Normalize z để nằm trong khoảng hợp lý
        zmin, zmax = z.min(), z.max()
        if zmax - zmin > 1e-6:
            # Scale z về khoảng [-1, 1] để dễ nhìn
            z = 2.0 * (z - zmin) / (zmax - zmin) - 1.0
            print(f"Z normalized to: [{z.min():.3f}, {z.max():.3f}]")
        else:
            print("Warning: z values are constant")

        # Gộp vertex (x, y, z)
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        self.vertices = vertices
        print(f"Created {len(vertices)} vertices")

        # Tạo index cho triangles
        indices = []
        for i in range(n - 1):
            for j in range(n - 1):
                top_left = i * n + j
                top_right = top_left + 1
                bottom_left = (i + 1) * n + j
                bottom_right = bottom_left + 1
                
                # Triangle 1
                indices.extend([top_left, bottom_left, top_right])
                # Triangle 2
                indices.extend([top_right, bottom_left, bottom_right])
                
        self.indices = np.array(indices, dtype=np.uint32)
        print(f"Created {len(self.indices)} indices ({len(self.indices)//3} triangles)")

        # Màu theo độ cao (gradient từ xanh -> đỏ)
        colors = np.zeros((len(vertices), 3), dtype=np.float32)
        
        # Normalize z cho color mapping
        z_flat = z.flatten()
        z_normalized = (z_flat - z_flat.min()) / (z_flat.max() - z_flat.min() + 1e-8)
        
        colors[:, 0] = z_normalized      # R: tăng theo độ cao
        colors[:, 1] = 0.5               # G: cố định
        colors[:, 2] = 1.0 - z_normalized  # B: giảm theo độ cao
        
        self.colors = colors

        # Shader & VAO
        self.shader = Shader(vert_shader, frag_shader)
        self.vao = VAO()
        self.uma = UManager(self.shader)
        self.transform = np.eye(4)  # Transform matrix

    def setup(self):
        """Setup VAO with vertex and color data"""
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
        print("EquationMesh setup complete")
        return self

    def draw(self, projection, view, model):
        """Draw the mesh"""
        GL.glUseProgram(self.shader.render_idx)
        
        if model is None:
            modelview = view
        else:
            modelview = view @ model
            
        self.uma.upload_uniform_matrix4fv(projection, "projection", True)
        self.uma.upload_uniform_matrix4fv(modelview, "modelview", True)
        
        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)