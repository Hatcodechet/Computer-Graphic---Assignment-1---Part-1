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
                 phong_vert, phong_frag,
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
        
        self.phong_vert = phong_vert
        self.phong_frag = phong_frag
        
        # Vibration mode
        self.vibration_mode = "All"

        # scene objects
        self.atoms = []   # list of dict {obj: Sphere, pos: np.array, element: "H/O/C", orig_pos: base position}
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

        # 1) Build atoms (colored spheres) with Phong shading
        for elem, pos in data["atoms"]:
            frag_path = self.elem_frag.get(elem, self.solid_gray)
            sphere = Sphere(self.phong_vert, self.phong_frag, stacks=24, slices=48).setup()
            # Set color based on element (use colors that work well with Phong shading)
            if elem == "H":
                sphere.set_color([0.9, 0.9, 0.9])  # Light gray-white for hydrogen
            elif elem == "O":
                sphere.set_color([0.8, 0.2, 0.2])  # Red for oxygen
            elif elem == "C":
                sphere.set_color([0.2, 0.2, 0.2])  # Dark gray for carbon (not pure black for better visibility)
            else:
                sphere.set_color([0.5, 0.5, 0.5])  # Gray for others
            pos_array = np.array(pos, dtype=np.float32)
            self.atoms.append({
                "obj": sphere,
                "pos": pos_array.copy(),  # Current animated position
                "orig_pos": pos_array.copy(),  # Original/base position
                "elem": elem
            })

        # 2) Build bonds (cylinders) with Phong shading and gray color
        for (ia, ib) in data["bonds"]:
            cyl = Cylinder(self.phong_vert, self.phong_frag, n=24, r_bottom=self.bond_radius, r_top=self.bond_radius).setup()
            # Set color to a subtle gray-brown for bonds (more realistic for molecular bonds)
            cyl.set_color([0.6, 0.55, 0.5])  # Subtle gray-brown for bonds
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
    
    def get_animated_positions(self):
        """Tính vị trí động của các nguyên tử dựa trên nhiều mode dao động."""
        if not self.animate:
            return [a["orig_pos"].copy() for a in self.atoms]

        t = self.time
        vib_mode = self.vibration_mode
        animated_positions = []
        
        # Get reference to oxygen for H2O
        o_pos = self.atoms[0]["orig_pos"] if self.molecule_name == "H2O" else None
        
        for i, a in enumerate(self.atoms):
            pos = a["orig_pos"].copy()
            elem = a["elem"]
            if i == 0:
                animated_positions.append(pos)
                continue
            
            # === STRETCHING MODES ===
            # Apply only if mode is "All" or "Stretching"
            if vib_mode in ["All", "Stretching"]:
                # For stretching: move along the bond direction only
                # In H2O: each H only stretches along its own O-H bond
                if self.molecule_name == "H2O" and elem == "H":
                    # H atoms: stretch along O-H bond
                    o_pos = self.atoms[0]["orig_pos"]
                    h_orig = self.atoms[i]["orig_pos"]
                    
                    # Direction from O to H
                    oh_vec = h_orig - o_pos
                    oh_len = np.linalg.norm(oh_vec)
                    
                    if oh_len > 1e-6:
                        oh_dir = oh_vec / oh_len
                        # First H and second H have different phases
                        freq = 2.5 if i == 1 else 3.0
                        phase = t * freq
                        stretch_amp = 0.04 * math.sin(phase)
                        # Stretch outward (away from O)
                        pos += oh_dir * stretch_amp
                else:
                    # For other molecules, apply bond stretching
                    for b_idx, bond in enumerate(self.bonds):
                        ai, bi = bond["a"], bond["b"]
                        
                        if i == ai or i == bi:
                            pa_orig = self.atoms[ai]["orig_pos"]
                            pb_orig = self.atoms[bi]["orig_pos"]
                            bond_vec = pb_orig - pa_orig
                            bond_len = np.linalg.norm(bond_vec)
                            
                            if bond_len > 1e-6:
                                bond_dir = bond_vec / bond_len
                                freq = 2.0 + b_idx * 1.5
                                phase = t * freq
                                
                                # Move along bond
                                direction = 1.0 if i == ai else -1.0
                                stretch_amp = 0.04 * math.sin(phase)
                                pos += bond_dir * stretch_amp * direction
                            break
            
            # Special H2O vibrational modes
            if self.molecule_name == "H2O" and elem == "H":
                # Determine which H atom (1st or 2nd hydrogen)
                is_first_h = (i == 1)
                h_sign = 1.0 if is_first_h else -1.0  # Opposite phase for symmetric modes
                
                # === BENDING (SCISSORING) ===
                if vib_mode in ["All", "Bending"] and o_pos is not None:
                    rel_vec = pos - o_pos
                    if np.linalg.norm(rel_vec) > 1e-6:
                        rel_dir = rel_vec / np.linalg.norm(rel_vec)
                        bend_amp = 0.08 * math.sin(t * 3.5)
                        pos += rel_dir * bend_amp * h_sign
                
                # === ROCKING ===
                if vib_mode in ["All", "Rocking"]:
                    rock_amp = 0.06 * math.sin(t * 1.8)
                    pos[0] += rock_amp
                
                # === WAGGING ===
                if vib_mode in ["All", "Wagging"]:
                    wag_amp = 0.07 * math.sin(t * 2.5)
                    pos[2] += wag_amp
                
                # === TWISTING ===
                if vib_mode in ["All", "Twisting"]:
                    twist_amp = 0.08 * math.sin(t * 4.2)
                    pos[2] += twist_amp * h_sign
            
            animated_positions.append(pos)
        
        return animated_positions



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
        # Lấy vị trí động (đã tính các mode dao động)
        animated_positions = self.get_animated_positions()
        
        # Nếu bật animation → quay nhẹ toàn phân tử quanh trục Y (hiệu ứng đẹp)
        #if self.animate:
            #rotation_matrix = rotate(axis=[0.0, 1.0, 0.0], angle=self.time * 20.0)
            #model = model @ rotation_matrix

        # =============================
        # 1️⃣ VẼ LIÊN KẾT (BONDS)
        # =============================
        for b in self.bonds:
            ai, bi = b["a"], b["b"]
            pa = animated_positions[ai]
            pb = animated_positions[bi]
            ea = self.atoms[ai]["elem"]
            eb = self.atoms[bi]["elem"]

            # Rút ngắn 2 đầu để cylinder chỉ chạm vỏ cầu
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

            # =============================
            # FIX: Giữ hướng bond theo hướng gốc
            # =============================
            orig_pa = self.atoms[ai]["orig_pos"]
            orig_pb = self.atoms[bi]["orig_pos"]

            # Hướng và chiều dài gốc
            orig_vec = orig_pb - orig_pa
            orig_len = np.linalg.norm(orig_vec)

            # Chiều dài hiện tại để co giãn
            curr_len = np.linalg.norm(pb2 - pa2)
            scale_factor = curr_len / orig_len if orig_len > 1e-6 else 1.0

            # Dựng bond theo hướng gốc, chỉ scale chiều dài
            M = self._bond_transform(orig_pa, orig_pb)
            S_fix = scale([1.0, scale_factor, 1.0])
            M = M @ S_fix

            b["obj"].draw(projection, view, model @ M)

        # =============================
        # 2️⃣ VẼ NGUYÊN TỬ (ATOMS)
        # =============================
        for i, a in enumerate(self.atoms):
            elem = a["elem"]
            r = self.atom_radius.get(elem, 0.2)
            anim_pos = animated_positions[i]
            M = translate(anim_pos) @ scale([r, r, r])
            a["obj"].draw(projection, view, model @ M)

