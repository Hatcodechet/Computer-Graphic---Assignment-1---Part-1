from OpenGL.GL import *
import OpenGL.GL as GL
import numpy as np
from PIL import Image
from tostudents.libs.shader import *
from tostudents.libs.buffer import *
import glfw


class ObjLoader:
    def __init__(self, filepath, vert_shader, frag_shader, texture_path=None):
        self.filepath = filepath
        self.vert_coords = []     # v
        self.text_coords = []     # vt

        self.vertex_index = []
        self.texture_index = []

        self.load()

        # --- Create VAO and Shader ---
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

        self.texture = None
        if texture_path:
            self.texture = self.load_texture(texture_path)

    # ------------------------------------------------------------
    # Load .OBJ file
    # ------------------------------------------------------------
    def load(self):
        for line in open(self.filepath, 'r'):
            if line.startswith('#'):
                continue
            values = line.strip().split()
            if not values:
                continue

            # Vertex
            if values[0] == 'v':
                self.vert_coords.append([float(v) for v in values[1:4]])

            # Texture coordinates
            elif values[0] == 'vt':
                self.text_coords.append([float(v) for v in values[1:3]])

            # elif values[0] == 'vn':
            #     self.norm_coords.append([float(v) for v in values[1:4]])

            # Face
            elif values[0] == 'f':
                face_i, text_i = [], []
                for v in values[1:4]:
                    w = v.split('/')
                    face_i.append(int(w[0]) - 1)
                    if len(w) >= 2 and w[1] != '':
                        text_i.append(int(w[1]) - 1)
                self.vertex_index.append(face_i)
                self.texture_index.append(text_i)

        # flat the index list   
        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]

        vertices = []
        texcoords = []

        for i in self.vertex_index:
            vertices.extend(self.vert_coords[i])
        for i in self.texture_index:
            texcoords.extend(self.text_coords[i])

        self.vertices = np.array(vertices, dtype='float32')
        self.texcoords = np.array(texcoords, dtype='float32')
        self.indices = np.array(range(len(self.vertex_index)), dtype='int32')

    # ------------------------------------------------------------
    # Load texture từ file ảnh
    # ------------------------------------------------------------
    def load_texture(self, path):
        img = Image.open(path)
        img_data = np.array(img.convert("RGBA"), dtype=np.uint8)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                        img.width, img.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        return tex

    # ------------------------------------------------------------
    # Setup GPU buffer
    # ------------------------------------------------------------
    def setup(self):
        # Vertex
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        # Texture coordinates
        self.vao.add_vbo(1, self.texcoords, ncomponents=2, stride=0, offset=None)
        # Indices
        self.vao.add_ebo(self.indices)
        return self

    # ------------------------------------------------------------
    # Draw object
    # ------------------------------------------------------------
    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if projection is not None:
            self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        if view is not None:
            self.uma.upload_uniform_matrix4fv(view, 'view', True)
        if model is not None:
            self.uma.upload_uniform_matrix4fv(model, 'model', True)

       

        #draw
        self.vao.activate()
        #GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


       ## GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
        #GL.glPolygonOffset(-1, -1)
        #GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        #GL.glLineWidth(0.3)
        #GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
       # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
       

