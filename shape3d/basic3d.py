from tostudents.libs.shader import *
from tostudents.libs import transform as T
from tostudents.libs.buffer import *
import numpy as np
import glfw
from OpenGL import GL
import ctypes
from tostudents.libs.transform import Trackball, translate, scale  
from PIL import Image


class Cube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = 0.5 * np.array(
            [
                [-1, -1, +1],  # A <= Bottom: ABCD
                [+1, -1, +1],  # B
                [+1, -1, -1],  # C
                [-1, -1, -1],  # D
                [-1, +1, +1],  # E <= Top: EFGH
                [+1, +1, +1],  # F
                [+1, +1, -1],  # G
                [-1, +1, -1],  # H
            ],
            dtype=np.float32
        )

        self.indices = np.array([0, 4, 1, 5, 2, 6, 3, 7, 0, 4, 4, 0, 0, 1, 3, 2,2 , 4, 4,5,7,6], dtype=np.int32)
        
        self.normals = self.vertices.copy()
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

        # colors: RGB format
        self.colors = np.array(
            [  # R    G    B
                [1.0, 0.0, 0.0],  # A <= Bottom: ABCD
                [1.0, 0.0, 1.0],  # B
                [0.0, 0.0, 1.0],  # C
                [0.0, 0.0, 0.0],  # D
                [1.0, 1.0, 0.0],  # E <= Top: EFGH
                [1.0, 1.0, 1.0],  # F
                [0.0, 1.0, 1.0],  # G
                [0.0, 1.0, 0.0],  # H
            ],
            dtype=np.float32
        )
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()


        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2
    
    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def generate_uv(self):
        self.texcoords = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],
            [0, 0], [1, 0], [1, 1], [0, 1],
        ], dtype=np.float32)

class Sphere(object):
    def __init__(self, vert_shader, frag_shader, stacks=32, slices=64, texture_path=None):
        self.stacks = stacks
        self.slices = slices
        self.radius = 0.5
        self.texture_path = texture_path

        # ----- 1. Tạo đỉnh -----
        vertices, normals, texcoords = [], [], []

        for i in range(stacks + 1):
            phi = np.pi * i / stacks  # 0 -> π
            y = np.cos(phi)
            r_stack = np.sin(phi)
            for j in range(slices + 1):
                theta = 2 * np.pi * j / slices  # 0 -> 2π
                x = r_stack * np.cos(theta)
                z = r_stack * np.sin(theta)
                vertices.append([x * self.radius, y * self.radius, z * self.radius])
                normals.append([x, y, z])
                texcoords.append([j / slices, 1 - i / stacks])  # UV chuẩn Earth

        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.texcoords = np.array(texcoords, dtype=np.float32)

        # ----- 2. Tạo indices dạng tam giác -----
        indices = []
        for i in range(stacks):
            for j in range(slices):
                first = i * (slices + 1) + j
                second = first + slices + 1
                indices += [first, second, first + 1, second, second + 1, first + 1]
        self.indices = np.array(indices, dtype=np.uint32)

        # ----- 3. Màu (chỉ để tránh lỗi shader nếu không có texture) -----
        self.colors = np.ones_like(self.vertices, dtype=np.float32)

        # ----- 4. Shader + VAO -----
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        # Texture
        self.texture_id = None
        if self.texture_path:
            self.load_texture(self.texture_path)

    # ---------------------------------------------------------
    def setup(self):
        """Setup VAO và buffer"""
        self.vao.add_vbo(0, self.vertices, ncomponents=3)
        self.vao.add_vbo(1, self.normals, ncomponents=3)
        self.vao.add_vbo(2, self.colors, ncomponents=3)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2)
        self.vao.add_ebo(self.indices)
        return self

    # ---------------------------------------------------------
    def load_texture(self, path):
        """Nạp texture từ file ảnh (Earth, Sun, v.v.)"""
        img = Image.open(path).convert("RGB")
        img_data = img.tobytes("raw", "RGB", 0, -1)

        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                        img.width, img.height, 0, GL.GL_RGB,
                        GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        print(f"[INFO] Texture loaded successfully: {path}")

    # ---------------------------------------------------------
    def draw(self, projection, view, model):
        """Vẽ sphere có texture"""
        GL.glUseProgram(self.shader.render_idx)
        modelview = view if model is None else view @ model

        self.uma.upload_uniform_matrix4fv(projection, "projection", True)
        self.uma.upload_uniform_matrix4fv(modelview, "modelview", True)

        if self.texture_id:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            tex_loc = GL.glGetUniformLocation(self.shader.render_idx, "tex")
            GL.glUniform1i(tex_loc, 0)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


class Cone(object):
    def __init__(self, vert_shader, frag_shader, slices=48):
        self.slices = slices
        r = 0.5
        h = 0.7

        vertices = []

        # (a) bottom (y = -h/2)
        for i in range(slices + 1):
            theta = 2 * np.pi * i / slices
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, -h, z])

        # (b) top
        vertices.append([0, +h / 2, 0])  # index = slices + 1

        # (c) bottom center
        vertices.append([0, -h / 2, 0])  # index = slices + 2

        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)


        # ----- 2. create indices -----
        indices = []
        apex_index = slices + 1
        center_bottom = slices + 2

        # (a) side (triangle strip)
        for i in range(slices + 1):
            indices += [i, apex_index]

        # degenerate 
        indices += [apex_index, center_bottom]

        # (b) bottom (triangle fan)
        for i in range(slices + 1):
            indices += [center_bottom, i]

        self.indices = np.array(indices, dtype=np.uint32)

       # ----- 3. color  -----
        colors = []
        for v in vertices:
            y = v[1]
            if y > 0.0:
                # phần gần đỉnh nón -> sáng hơn
                colors.append([1.0, 1.0 - y, 1.0 - y * 0.5])
            elif abs(y + 0.35) < 0.01:
                # tâm đáy → màu xanh lam
                colors.append([0.2, 0.6, 1.0])
            else:
                # mặt bên gần đáy → chuyển đỏ sang vàng
                t = (y + 0.35) / 0.7
                colors.append([1.0, t, 0.2])
        self.colors = np.array(colors, dtype=np.float32)
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()


        # ----- 4. Shader + VAO -----
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    # ---------------------------------------------------------
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)

        self.vao.add_ebo(self.indices)
        return self

    # ---------------------------------------------------------
    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def generate_uv(self):
        uv = []
        for i in range(self.slices + 1):
            u = i / self.slices
            uv.append([u, 0])
        uv.append([0.5, 1.0])  # apex
        uv.append([0.5, 0.5])  # bottom center
        self.texcoords = np.array(uv, dtype=np.float32)


    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

class ConeFan(object):
    def __init__(self, vert_shader, frag_shader, slices=48):
        self.slices = slices
        r = 0.5
        h = 0.7

        # ----- 1. Tạo đỉnh -----
        vertices = []

        # (a) vòng tròn đáy (y = -h/2)
        for i in range(slices + 1):
            theta = 2 * np.pi * i / slices
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, -h/2, z])

        # (b) đỉnh nón
        vertices.append([0, +h / 2, 0])  # index = slices + 1

        # (c) tâm đáy
        vertices.append([0, -h / 2, 0])  # index = slices + 2

        self.vertices = np.array(vertices, dtype=np.float32)
        # --- Normal vectors ---
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)


        # ----- 2.  indice side from top center -----
        side_indices = [slices + 1]  
        for i in range(slices + 1):
            side_indices.append(i)
        
        self.side_indices = np.array(side_indices, dtype=np.uint32)

        # ----- 3. Tạo indices CHO ĐÁY (fan từ tâm đáy) -----
        bottom_indices = [slices + 2]  # bắt đầu từ tâm đáy
        for i in range(slices, -1, -1):  # ngược chiều để mặt hướng đúng
            bottom_indices.append(i)
        
        self.bottom_indices = np.array(bottom_indices, dtype=np.uint32)

        # ----- 4. Màu -----
        colors = []
        for v in vertices:
            y = v[1]
            if y > 0.0:
                colors.append([1.0, 1.0 - y, 1.0 - y * 0.5])
            elif abs(y + h/2) < 0.01:
                colors.append([0.2, 0.6, 1.0])
            else:
                t = (y + h/2) / h
                colors.append([1.0, t, 0.2])
        self.colors = np.array(colors, dtype=np.float32)

        # ----- 5. Shader + VAO -----
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)


        # Chỉ cần add EBO một lần (có thể dùng side hoặc bottom, hoặc concatenate)
        # Nhưng để linh động, ta sẽ bind thủ công trong draw()
        self.vao.add_ebo(self.side_indices)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Vẽ mặt bên
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vao.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.side_indices.nbytes, 
                        self.side_indices, GL.GL_STATIC_DRAW)
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, len(self.side_indices), 
                         GL.GL_UNSIGNED_INT, None)

        # Vẽ đáy
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.bottom_indices.nbytes, 
                        self.bottom_indices, GL.GL_STATIC_DRAW)
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, len(self.bottom_indices), 
                         GL.GL_UNSIGNED_INT, None)
        
    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

class Cylinder(object):
    def __init__(self, vert_shader, frag_shader, n=32):
        self.n = n
        r = 0.5
        h = 0.5

        # ----- 1. Tạo đỉnh -----
        vertices = []

        # (a) vòng tròn dưới (y = -h)
        for i in range(n):
            theta = 2 * np.pi * i / n
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, -h, z])

        # (b) vòng tròn trên (y = +h)
        for i in range(n):
            theta = 2 * np.pi * i / n
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, +h, z])

        # (c) tâm đáy dưới & tâm đáy trên
        vertices.append([0, -h, 0])  # index 2n
        vertices.append([0, +h, 0])  # index 2n+1

        self.vertices = np.array(vertices, dtype=np.float32)
        # --- Normal vectors ---
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)


        # ----- 2. Tạo indices -----
        indices = []

        # (a) Thân hình trụ (dùng triangle strip)
        for i in range(n):
            indices += [i, (i + n), (i + 1) % n, ((i + 1) % n) + n]

        # (b) triangle fan_ bottom
        center_bottom = 2 * n
        for i in range(n):
            indices += [center_bottom, (i + 1) % n, i]

       # # (c) triangle fan top
        center_top = 2 * n + 1
        for i in range(n):
            indices += [center_top, i + n, ((i + 1) % n) + n]

        self.indices = np.array(indices, dtype=np.uint32)

        # ----- 3. color -----
        colors = []
        for i in range(len(vertices)):
            colors.append([
                0.5 + 0.5 * np.cos(i), 
                0.5 + 0.5 * np.sin(i), 
                1.0
            ])
        self.colors = np.array(colors, dtype=np.float32)
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()

        # ----- 4. Shader + VAO -----
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    # ---------------------------------------------------------
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)


        self.vao.add_ebo(self.indices)
        return self

    # ---------------------------------------------------------
    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        if self.texture_id:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            self.uma.upload_uniform_1i(0, "tex")

        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def generate_uv(self):
        n = self.n
        uv = []
        for i in range(n):
            u = i / n
            uv.append([u, 0])
        for i in range(n):
            u = i / n
            uv.append([u, 1])
        uv += [[0.5, 0.5], [0.5, 0.5]]
        self.texcoords = np.array(uv, dtype=np.float32)

'''
class ConeWithCylinder(object):
    """Composite object: Cone trên, Cylinder dưới"""
    def __init__(self, vert_shader, frag_shader, slices=48):
        # Tạo cone và cylinder riêng biệt
        self.cone = ConeFan(vert_shader, frag_shader, slices)
        self.cylinder = Cylinder(vert_shader, frag_shader, slices)
        
    def setup(self):
        self.cone.setup()
        self.cylinder.setup()
        return self
    
    def draw(self, projection, view, model):
        # Nếu có model transform từ bên ngoài, apply vào base transform
        if model is None:
            model = np.identity(4, dtype=np.float32)
        
        # Vẽ cone ở phía trên
        cone_model = model @ translate(0, 0.5, 0) 
        self.cone.draw(projection, view, cone_model)
        
        # Vẽ cylinder ở phía dưới, nối liền với cone
        cylinder_model = model @ translate(0, -0.35, 0) 
        self.cylinder.draw(projection, view, cylinder_model)

    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
'''

class Tetrahedron(object):
    def __init__(self, vert_shader, frag_shader, size=1.0):
        """
        Tứ diện đều (Regular Tetrahedron)
        size: độ dài cạnh
        """
        self.size = size
        
        # Tọa độ đơn giản hơn: tứ diện đều với cạnh = size
        # Đặt tâm tại gốc tọa độ
        s = size / np.sqrt(2)  # = 0.7071 * size
        
        # 4 đỉnh tạo thành tứ diện đều
        self.vertices = np.array([
            [s,   s,   s],    # Đỉnh 0: (+, +, +)
            [s,  -s,  -s],    # Đỉnh 1: (+, -, -)
            [-s,  s,  -s],    # Đỉnh 2: (-, +, -)
            [-s, -s,   s],    # Đỉnh 3: (-, -, +)
        ], dtype=np.float32)
        
        #  (size=1.0):
        # self.vertices = np.array([
        #     [0.7071,   0.7071,   0.7071],   # Đỉnh 0
        #     [0.7071,  -0.7071,  -0.7071],   # Đỉnh 1
        #     [-0.7071,  0.7071,  -0.7071],   # Đỉnh 2
        #     [-0.7071, -0.7071,   0.7071],   # Đỉnh 3
        # ], dtype=np.float32)
        
        # Indices cho 4 mặt tam giác
        self.indices = np.array([
            0, 1, 2,  # Mặt 1
            0, 2, 3,  # Mặt 2
            0, 3, 1,  # Mặt 3
            1, 3, 2,  # Mặt 4 (đáy)
        ], dtype=np.uint32)
        
        # Màu sắc cho mỗi đỉnh
        self.colors = np.array([
            [1.0, 0.0, 0.0],  # Đỏ
            [0.0, 1.0, 0.0],  # Xanh lá
            [0.0, 0.0, 1.0],  # Xanh dương
            [1.0, 1.0, 0.0],  # Vàng
        ], dtype=np.float32)
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()
        
        # Shader & VAO
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.transform = np.eye(4)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)


        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        if self.texture_id:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            self.uma.upload_uniform_1i(0, "tex")

        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
    def generate_uv(self):
        self.texcoords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.5],
        ], dtype=np.float32)


class Cylinder2(object):
    def __init__(self, vert_shader, frag_shader, n=32, r_bottom=0.3, r_top=0.1):
        """
        Truncated Cone (Hình nón cụt)
        
        Args:
            n: số đỉnh trên mỗi vòng tròn
            r_bottom: 
            r_top: b
        """
        self.n = n
        h = 0.5  # chiều cao

        # ----- 1. Tạo đỉnh -----
        vertices = []

        # (a) vòng tròn dưới (y = -h, bán kính = r_bottom)
        for i in range(n):
            theta = 2 * np.pi * i / n
            x = r_bottom * np.cos(theta)
            z = r_bottom * np.sin(theta)
            vertices.append([x, -h, z])

        # (b) vòng tròn trên (y = +h, bán kính = r_top)
        for i in range(n):
            theta = 2 * np.pi * i / n
            x = r_top * np.cos(theta)
            z = r_top * np.sin(theta)
            vertices.append([x, +h, z])

        # (c) tâm đáy dưới & tâm đáy trên
        vertices.append([0, -h, 0])  # index 2n
        vertices.append([0, +h, 0])  # index 2n+1

        self.vertices = np.array(vertices, dtype=np.float32)
        # --- Normal vectors ---
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)


        # ----- 2. Tạo indices -----
        indices = []

        # (a) Thân hình nón cụt (dùng triangles thay vì triangle strip)
        for i in range(n):
            next_i = (i + 1) % n
            # Triangle 1: bottom[i] -> top[i] -> bottom[next_i]
            indices += [i, i + n, next_i]
            # Triangle 2: bottom[next_i] -> top[i] -> top[next_i]
            indices += [next_i, i + n, next_i + n]

        # (b) Đáy dưới (triangle fan)
        center_bottom = 2 * n
        for i in range(n):
            indices += [center_bottom, (i + 1) % n, i]

        # (c) Đáy trên (triangle fan)
        center_top = 2 * n + 1
        for i in range(n):
            indices += [center_top, i + n, ((i + 1) % n) + n]

        self.indices = np.array(indices, dtype=np.uint32)

        # ----- 3. Màu -----
        colors = []
        for i in range(len(vertices)):
            colors.append([
                0.5 + 0.5 * np.cos(i), 
                0.5 + 0.5 * np.sin(i), 
                1.0
            ])
        self.colors = np.array(colors, dtype=np.float32)
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()

        # ----- 4. Shader + VAO -----
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.transform = np.eye(4)  # Thêm transform matrix

    # ---------------------------------------------------------
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)

        self.vao.add_ebo(self.indices)
        return self

    # ---------------------------------------------------------
    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
    def generate_uv(self):
        n = self.n
        uv = []
        for i in range(n):
            u = i / n
            uv.append([u, 0])
        for i in range(n):
            u = i / n
            uv.append([u, 1])
        uv += [[0.5, 0.5], [0.5, 0.5]]  # center top/bottom
        self.texcoords = np.array(uv, dtype=np.float32)

    def load_texture(self, path):
        """Load ảnh ngoài vào OpenGL texture"""
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
        img_data = np.array(img, dtype=np.uint8)
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)


class Torus(object):
    def __init__(self, vert_shader, frag_shader, major_segments=32, minor_segments=16, major_radius=0.4, minor_radius=0.15):
        """
        Torus (Hình xuyến - hình bánh donut)
        
        Args:
            major_segments: số đoạn trên vòng tròn lớn (xung quanh tâm)
            minor_segments: số đoạn trên vòng tròn nhỏ (ống tròn)
            major_radius: bán kính vòng tròn lớn (R - từ tâm đến tâm ống)
            minor_radius: bán kính vòng tròn nhỏ (r - bán kính của ống)
        """
        self.major_segments = major_segments
        self.minor_segments = minor_segments
        R = major_radius  # Bán kính lớn
        r = minor_radius  # Bán kính nhỏ

        # ----- 1. Tạo đỉnh -----
        vertices = []
        
        for i in range(major_segments):
            theta = 2 * np.pi * i / major_segments  # góc quay quanh trục Y
            
            for j in range(minor_segments):
                phi = 2 * np.pi * j / minor_segments  # góc quay của ống tròn
                
                # Công thức tọa độ torus
                x = (R + r * np.cos(phi)) * np.cos(theta)
                y = r * np.sin(phi)
                z = (R + r * np.cos(phi)) * np.sin(theta)
                
                vertices.append([x, y, z])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        # --- Normal vectors ---
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)


        # ----- 2. Tạo indices (sử dụng triangles) -----
        indices = []
        
        for i in range(major_segments):
            for j in range(minor_segments):
                # Tính chỉ số 4 đỉnh của quad
                current = i * minor_segments + j
                next_i = ((i + 1) % major_segments) * minor_segments + j
                next_j = i * minor_segments + (j + 1) % minor_segments
                next_both = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
                
                # Chia quad thành 2 tam giác
                indices += [current, next_i, next_j]
                indices += [next_j, next_i, next_both]
        
        self.indices = np.array(indices, dtype=np.uint32)

        # ----- 3. Màu (gradient dựa trên vị trí) -----
        colors = []
        for i, v in enumerate(vertices):
            # Tạo màu gradient đẹp mắt
            t1 = i / len(vertices)
            colors.append([
                0.3 + 0.7 * np.sin(t1 * np.pi * 2),
                0.5 + 0.5 * np.cos(t1 * np.pi * 3),
                0.8
            ])
        self.colors = np.array(colors, dtype=np.float32)
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()

        # ----- 4. Shader + VAO -----
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)


        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def generate_uv(self):
        uv = []
        for i in range(self.major_segments):
            for j in range(self.minor_segments):
                u = i / self.major_segments
                v = j / self.minor_segments
                uv.append([u, v])
        self.texcoords = np.array(uv, dtype=np.float32)
    def load_texture(self, path):
        """Load ảnh ngoài vào OpenGL texture"""
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
        img_data = np.array(img, dtype=np.uint8)
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)



class Prism(object):
    def __init__(self, vert_shader, frag_shader, n_sides=6, height=0.8, radius=0.4):
        """
        Prism (Lăng trụ đều)
        
        Args:
            n_sides: số cạnh của đa giác đáy (3=tam giác, 4=vuông, 5=ngũ giác, 6=lục giác,...)
            height: chiều cao của lăng trụ
            radius: bán kính đường tròn ngoại tiếp đa giác đáy
        """
        self.n_sides = n_sides
        h = height / 2  # chia đôi để đối xứng qua trục
        r = radius

        # ----- 1. Tạo đỉnh -----
        vertices = []

        # (a) Đa giác đáy dưới (y = -h)
        for i in range(n_sides):
            theta = 2 * np.pi * i / n_sides
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, -h, z])

        # (b) Đa giác đáy trên (y = +h)
        for i in range(n_sides):
            theta = 2 * np.pi * i / n_sides
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, +h, z])

        # (c) Tâm đáy dưới và tâm đáy trên (để vẽ đáy bằng triangle fan)
        vertices.append([0, -h, 0])  # index: 2*n_sides
        vertices.append([0, +h, 0])  # index: 2*n_sides + 1

        self.vertices = np.array(vertices, dtype=np.float32)
        # --- Normal vectors ---
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)


        # ----- 2. Tạo indices -----
        indices = []

        # (a) Các mặt bên (side faces)
        for i in range(n_sides):
            next_i = (i + 1) % n_sides
            
            # Triangle 1: bottom[i] -> top[i] -> bottom[next_i]
            indices += [i, i + n_sides, next_i]
            
            # Triangle 2: bottom[next_i] -> top[i] -> top[next_i]
            indices += [next_i, i + n_sides, next_i + n_sides]

        # (b) Đáy dưới (triangle fan từ tâm)
        center_bottom = 2 * n_sides
        for i in range(n_sides):
            indices += [center_bottom, (i + 1) % n_sides, i]

        # (c) Đáy trên (triangle fan từ tâm)
        center_top = 2 * n_sides + 1
        for i in range(n_sides):
            indices += [center_top, i + n_sides, ((i + 1) % n_sides) + n_sides]

        self.indices = np.array(indices, dtype=np.uint32)

        # ----- 3. Màu -----
        colors = []
        for i in range(len(vertices)):
            # Tạo màu gradient theo vị trí
            if i < n_sides:  # Đáy dưới - xanh lam đậm
                colors.append([0.2, 0.4, 0.8])
            elif i < 2 * n_sides:  # Đáy trên - xanh lam nhạt
                colors.append([0.4, 0.7, 1.0])
            else:  # Tâm - vàng
                colors.append([1.0, 0.9, 0.3])
        
        self.colors = np.array(colors, dtype=np.float32)
        # ---- Texture setup ----
        self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        self.texture_id = None
        self.generate_uv()

        # ----- 4. Shader + VAO -----
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)

        self.vao.add_ebo(self.indices)
        return self
    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            modelview = view
        else:
            modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
    def set_color(self, rgb):
        """Cập nhật màu Flat từ Viewer"""
        rgb = np.array(rgb, dtype=np.float32)
        self.colors = np.tile(rgb, (self.vertices.shape[0], 1))
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def generate_uv(self):
        n = self.n_sides
        uv = []
        for i in range(n):
            u = i / n
            uv.append([u, 0])
        for i in range(n):
            u = i / n
            uv.append([u, 1])
        uv += [[0.5, 0.5], [0.5, 0.5]]
        self.texcoords = np.array(uv, dtype=np.float32)

    def load_texture(self, path):
        """Load ảnh ngoài vào OpenGL texture"""
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
        img_data = np.array(img, dtype=np.uint8)
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)


