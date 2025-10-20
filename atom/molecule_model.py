import math
import numpy as np
import OpenGL.GL as GL
from tostudents.libs.transform import translate, scale, rotate
from tostudents.shape3d.basic3d import Sphere
# Nếu cylinder của bạn tên khác, sửa lại import sau:
from tostudents.shape3d.basic3d import Cylinder2 as Cylinder  # <-- đổi về Cylinder nếu cần


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def _rotation_from_to(frm, to):
    """Trả về (axis, angle) để quay vector frm về vector to"""
    frm = _normalize(np.array(frm, dtype=np.float32))
    to = _normalize(np.array(to, dtype=np.float32))
    c = np.clip(np.dot(frm, to), -1.0, 1.0)
    if c > 0.9999:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0
    if c < -0.9999:
        # đối hướng: quay 180 độ quanh 1 trục vuông góc bất kỳ
        axis = _normalize(np.cross(frm, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-6:
            axis = _normalize(np.cross(frm, np.array([0.0, 1.0, 0.0])))
        return axis, math.pi
    axis = _normalize(np.cross(frm, to))
    angle = math.acos(c)
    return axis, angle


class MoleculeModel:
    def __init__(self,
                 molecule_name,
                 solid_vert, solid_white, solid_red, solid_black, solid_gray,
                 gouraud_vert, gouraud_frag,
                 animate=False):
        self.molecule_name = molecule_name
        self.animate = animate

        # shader paths
        self.solid_vert = solid_vert
        self.solid_white = solid_white
        self.solid_red = solid_red
        self.solid_black = solid_black
        self.solid_gray = solid_gray

        self.gouraud_vert = gouraud_vert
        self.gouraud_frag = gouraud_frag

        # scene objects
        self.atoms = []   # list of dict {obj: Sphere, pos: np.array, element: "H/O/C"}
        self.bonds = []   # list of dict {obj: Cylinder, a: idx, b: idx}

        # Radii for drawing (not real covalent radii)
        self.atom_radius = {"H": 0.1, "O": 0.1, "C": 0.35}
        self.bond_radius = 0.06

        # Predefined molecules (approx positions)
        self.molecules = {
            "H2O": {
                "atoms": [("O", [0.0, 0.0, 0.0]),
                          ("H", [0.96, 0.26, 0.0]),
                          ("H", [-0.96, 0.26, 0.0])],
                "bonds": [(0, 1), (0, 2)]
            },
            "CO2": {
                "atoms": [("C", [0.0, 0.0, 0.0]),
                          ("O", [1.16, 0.0, 0.0]),
                          ("O", [-1.16, 0.0, 0.0])],
                "bonds": [(0, 1), (0, 2)]
            }
        }

        # Map element -> shader fragment (solid color)
        self.elem_frag = {
            "H": self.solid_white,
            "O": self.solid_red,
            "C": self.solid_black,
        }

        self.time = 0.0

    def setup(self):
        data = self.molecules.get(self.molecule_name)
        if self.molecule_name == "H2O":
            half = math.radians(104.5 / 2.0)  # ~52.25°
            dist = 0.90                       # bond length ngắn lại
            o = [0.0, 0.0, 0.0]
            h1 = [math.sin(half) * dist, math.cos(half) * dist, 0.0]
            h2 = [-math.sin(half) * dist, math.cos(half) * dist, 0.0]
            data = {
                "atoms": [("O", o), ("H", h1), ("H", h2)],
                "bonds": [(0, 1), (0, 2)]
            }

        if not data:
            raise ValueError(f"Unknown molecule: {self.molecule_name}")

        self.atoms.clear()
        self.bonds.clear()

        # 1) Build atoms (colored spheres)
        for elem, pos in data["atoms"]:
            frag_path = self.elem_frag.get(elem, self.solid_gray)
            sphere = Sphere(self.gouraud_vert, self.gouraud_frag, stacks=24, slices=48).setup()
            self.atoms.append({
                "obj": sphere,
                "pos": np.array(pos, dtype=np.float32),
                "elem": elem
            })

        # 2) Build bonds (cylinders)
        for (ia, ib) in data["bonds"]:
            cyl = Cylinder(self.gouraud_vert, self.gouraud_frag, n=24, r_bottom=self.bond_radius, r_top=self.bond_radius).setup()
            self.bonds.append({
                "obj": cyl,
                "a": ia,
                "b": ib
            })
        return self

    def update(self, dt):
        # Optional: molecule rotation/vibration
        if self.animate:
            self.time += dt

    def _bond_transform(self, pa, pb):
        """Đặt cylinder (mặc định dựng dọc +Y, cao = 1) nối từ pa đến pb.
        Trả về model = T * R * S với:
        - S: scale Y = length, X/Z = bond_radius
        - R: ma trận quay sao cho +Y local trùng hướng (pb - pa)
        - T: tịnh tiến về midpoint
        """
        pa = np.array(pa, dtype=np.float32)
        pb = np.array(pb, dtype=np.float32)
        v = pb - pa
        L = float(np.linalg.norm(v))
        mid = (pa + pb) * 0.5

        # Trường hợp trùng điểm (phòng thủ)
        if L < 1e-8:
            S = np.diag([self.bond_radius, 1.0, self.bond_radius, 1.0]).astype(np.float32)
            T = np.eye(4, dtype=np.float32); T[:3, 3] = mid
            return T @ S

        # Hướng đích (muốn +Y local trùng với hướng này)
        y = v / L  # (3,)

        # Chọn vector tạm để tạo cơ sở trực chuẩn (tránh song song)
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Tạo cơ sở trực chuẩn: x, y, z
        x = np.cross(tmp, y);  nx = np.linalg.norm(x)
        if nx < 1e-8:  # fallback nếu hiếm khi bị thẳng hàng
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            x = np.cross(tmp, y); nx = np.linalg.norm(x)
        x = x / nx
        z = np.cross(y, x)
        # Ma trận quay: cột (hoặc hàng) là các trục local -> world
        R = np.eye(4, dtype=np.float32)
        R[:3, 0] = x   # local +X -> world x
        R[:3, 1] = y   # local +Y -> world y (theo v)
        R[:3, 2] = z   # local +Z -> world z

        # Scale: X,Z = bán kính; Y = chiều dài
        S = np.diag([self.bond_radius, L, self.bond_radius, 1.0]).astype(np.float32)

        # Translate tới midpoint
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = mid

        # Model matrix
        return T @ R @ S


    def draw(self, projection, view, model):
        # Optional global transform for molecule animation
        if self.animate:
            # small rotation around Y
            model = model @ rotate(self.time * 20.0, [0.0, 1.0, 0.0])

        # 1) Draw bonds (behind) — cylinders
        for b in self.bonds:
            ai, bi = b["a"], b["b"]
            pa = self.atoms[ai]["pos"]; ea = self.atoms[ai]["elem"]
            pb = self.atoms[bi]["pos"]; eb = self.atoms[bi]["elem"]

            # rút ngắn 2 đầu theo bán kính atom để cylinder chỉ chạm vỏ cầu
            ra = self.atom_radius.get(ea, 0.2)
            rb = self.atom_radius.get(eb, 0.2)

            v = pb - pa
            L = float(np.linalg.norm(v))
            if L > 1e-6:
                d = v / L
                pa2 = pa + d * ra
                pb2 = pb - d * rb
            else:
                pa2, pb2 = pa, pb

            M = self._bond_transform(pa2, pb2)
            b["obj"].draw(projection, view, model @ M)


        # 2) Draw atoms — spheres with per-element radius
        for a in self.atoms:
            elem = a["elem"]
            r = self.atom_radius.get(elem, 0.2)
            M = translate(a["pos"]) @ scale([r, r, r])
            a["obj"].draw(projection, view, model @ M)
