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
        self.atom_radius = {"H": 0.18, "O": 0.25, "C": 0.22}
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
        if not data:
            raise ValueError(f"Unknown molecule: {self.molecule_name}")

        self.atoms.clear()
        self.bonds.clear()

        # 1) Build atoms (colored spheres)
        for elem, pos in data["atoms"]:
            frag_path = self.elem_frag.get(elem, self.solid_gray)
            sphere = Sphere(self.solid_vert, frag_path, stacks=24, slices=48).setup()
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
        """Trả về ma trận model để đặt cylinder dọc theo đoạn thẳng pa->pb (mặc định cylinder của bạn dựng theo trục +Y hoặc +Z?).
           Ở đây giả sử cylinder mặc định dựng dọc trục +Y, cao = 1.
           -> scale theo chiều Y = length, xoay từ +Y về hướng (pb-pa), translate tới midpoint.
        """
        pa = np.array(pa, dtype=np.float32)
        pb = np.array(pb, dtype=np.float32)
        mid = (pa + pb) / 2.0
        vec = pb - pa
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return translate(mid) @ scale([self.bond_radius, 1.0, self.bond_radius])

        direction = vec / length
        # +Y axis
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis, angle = _rotation_from_to(y_axis, direction)

        # scale XZ = radius, Y = length
        S = scale([1.0, length, 1.0])
        # rotate around 'axis' by 'angle' (axis-angle -> your rotate(ax, angle) expects degrees?)
        # Assuming rotate(angle, axis) from your utils is rotate(angle_in_degrees, axis_xyz)
        R = rotate(np.degrees(angle), axis)
        T = translate(mid)
        return T @ R @ S

    def draw(self, projection, view, model):
        # Optional global transform for molecule animation
        if self.animate:
            # small rotation around Y
            model = model @ rotate(self.time * 20.0, [0.0, 1.0, 0.0])

        # 1) Draw bonds (behind) — cylinders
        for b in self.bonds:
            pa = self.atoms[b["a"]]["pos"]
            pb = self.atoms[b["b"]]["pos"]
            M = self._bond_transform(pa, pb)
            b["obj"].draw(projection, view, model @ M)

        # 2) Draw atoms — spheres with per-element radius
        for a in self.atoms:
            elem = a["elem"]
            r = self.atom_radius.get(elem, 0.2)
            M = translate(a["pos"]) @ scale([r, r, r])
            a["obj"].draw(projection, view, model @ M)
