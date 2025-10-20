import math
import numpy as np
import OpenGL.GL as GL
from tostudents.libs.transform import translate, scale
from tostudents.shape3d.basic3d import Sphere


class AtomModel:
    def __init__(self, atom_name, vert_shader, frag_shader, animate=False):
        self.atom_name = atom_name
        self.vert_shader = vert_shader          # shader của Sphere (nucleus & electrons)
        self.frag_shader = frag_shader
        self.animate = animate

        # Spheres
        self.nucleus = None
        self.electrons = []                     # [{ "obj": Sphere, "radius": float, "angle": float, "speed": float }]

        # Orbit (VAO/VBO + shader riêng)
        self.orbits = []                        # list[(vao, count)]
        self.orbit_shader = None                # program id

        # Thời gian/animation
        self.time = 0.0

        # Cấu hình electron (Bohr, tối giản để hiển thị)
        self.shell_config = {
            "Hydrogen": [1],
            "Helium":   [2],
            "Carbon":   [2, 4],
            "Oxygen":   [2, 6],
            "Neon":     [2, 8],
            "Sodium":   [2, 8, 1],
        }

        # Tỉ lệ hiển thị
        self.nucleus_radius = 0.18              # scale khi vẽ nucleus
        self.electron_radius = 0.05             # scale khi vẽ electron

    # ---------------------------------------------------------
    # SETUP: tạo nucleus + electrons + orbit VAO/VBO
    # ---------------------------------------------------------
    def setup(self):
        # 1) Nucleus (sphere dùng shader chung của bạn)
        self.nucleus = Sphere(self.vert_shader, self.frag_shader, stacks=32, slices=64).setup()

        # 2) Shader riêng để vẽ orbit trắng
        self.orbit_shader = self._load_shader(
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/orbit.vert",
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/orbit.frag"
        )

        # 3) Xây dựng electrons + orbits theo shell
        self.electrons.clear()
        self.orbits.clear()

        shells = self.shell_config.get(self.atom_name, [1])
        for n, num_elec in enumerate(shells):
            # bán kính quỹ đạo hiển thị
            radius = 0.5 + n * 0.3

            # 3.a Orbit VAO/VBO cho shell n
            vertices = []
            segments = 100
            for i in range(segments):
                theta = 2 * math.pi * i / segments
                vertices += [radius * math.cos(theta), 0.0, radius * math.sin(theta)]
            vertices = np.array(vertices, dtype=np.float32)

            vao = GL.glGenVertexArrays(1)
            vbo = GL.glGenBuffers(1)
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)
            GL.glBindVertexArray(0)
            self.orbits.append((vao, len(vertices) // 3))

            # 3.b Electrons trên shell n (đặt đều theo góc)
            for i in range(num_elec):
                angle = (i / max(1, num_elec)) * 2 * math.pi
                speed = 1.0 / (n + 1)          # shell ngoài quay chậm hơn chút
                e_sphere = Sphere(self.vert_shader, self.frag_shader, stacks=16, slices=32).setup()
                self.electrons.append({
                    "obj": e_sphere,
                    "radius": radius,
                    "angle": angle,
                    "speed": speed,
                })

        return self

    # ---------------------------------------------------------
    # LOAD orbit shader (từ 2 file .vert / .frag)
    # ---------------------------------------------------------
    def _load_shader(self, vert_path, frag_path):
        with open(vert_path, "r") as vf:
            vert_src = vf.read()
        with open(frag_path, "r") as ff:
            frag_src = ff.read()

        vert = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert, vert_src)
        GL.glCompileShader(vert)
        if not GL.glGetShaderiv(vert, GL.GL_COMPILE_STATUS):
            raise RuntimeError(f"VERTEX SHADER ERROR:\n{GL.glGetShaderInfoLog(vert).decode()}")

        frag = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag, frag_src)
        GL.glCompileShader(frag)
        if not GL.glGetShaderiv(frag, GL.GL_COMPILE_STATUS):
            raise RuntimeError(f"FRAGMENT SHADER ERROR:\n{GL.glGetShaderInfoLog(frag).decode()}")

        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vert)
        GL.glAttachShader(prog, frag)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(f"PROGRAM LINK ERROR:\n{GL.glGetProgramInfoLog(prog).decode()}")

        # shader object có thể xoá sau khi link
        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)
        return prog

    # ---------------------------------------------------------
    # UPDATE: quay electron quanh nucleus
    # ---------------------------------------------------------
    def update(self, delta_time):
        if not self.animate:
            return
        for e in self.electrons:
            e["angle"] += e["speed"] * delta_time

    # ---------------------------------------------------------
    # DRAW: vẽ orbit (màu trắng), nucleus, rồi electrons
    # ---------------------------------------------------------
    def draw(self, projection, view, model):
        # 1) ORBITS (đường trắng) — dùng shader riêng
        GL.glUseProgram(self.orbit_shader)
        loc_p = GL.glGetUniformLocation(self.orbit_shader, "projection")
        loc_v = GL.glGetUniformLocation(self.orbit_shader, "view")
        loc_m = GL.glGetUniformLocation(self.orbit_shader, "model")
        # NumPy row-major → dùng transpose=True
        GL.glUniformMatrix4fv(loc_p, 1, GL.GL_TRUE, projection)
        GL.glUniformMatrix4fv(loc_v, 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(loc_m, 1, GL.GL_TRUE, model)

        GL.glLineWidth(1.0)  # macOS: nên để 1.0
        # nếu muốn orbit nổi lên trên, có thể tắt depth khi vẽ orbit:
        # GL.glDisable(GL.GL_DEPTH_TEST)
        for vao, count in self.orbits:
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, count)
        GL.glBindVertexArray(0)
        # GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)

        # 2) NUCLEUS — sphere trung tâm
        nucleus_model = model @ scale([self.nucleus_radius] * 3)
        self.nucleus.draw(projection, view, nucleus_model)

        # 3) ELECTRONS — spheres nhỏ chạy trên các quỹ đạo
        for e in self.electrons:
            r = e["radius"]
            a = e["angle"]
            x, z = r * math.cos(a), r * math.sin(a)
            e_model = model @ translate([x, 0.0, z]) @ scale([self.electron_radius] * 3)
            e["obj"].draw(projection, view, e_model)
