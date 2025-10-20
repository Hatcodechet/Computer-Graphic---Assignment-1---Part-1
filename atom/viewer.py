import OpenGL.GL as GL
import glfw
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer

from tostudents.libs.transform import Trackball
from tostudents.atom.atom_model import AtomModel
from tostudents.atom.molecule_model import *
from tostudents.main.axes import Axes


# ==========================================
# UI STATE
# ==========================================
class UIState:
    def __init__(self):
        # Part selector
        self.modes = ["Atom (Bohr)", "Molecule (Ball-and-Stick)"]
        self.mode_idx = 0

        # Atom (Bohr)
        self.atoms = ["Hydrogen", "Helium", "Carbon", "Oxygen", "Neon", "Sodium"]
        self.atom_idx = 0
        self.animate_electron = True

        # Molecule (Ball-and-Stick)
        self.molecules = ["H2O", "CO2"]
        self.mol_idx = 0
        self.animate_molecule = False  # (optional) rung/rotate

        # Common
        self.show_axes = True


# ==========================================
# VIEWER CLASS
# ==========================================
class Viewer:
    def __init__(self, width=1200, height=900):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.win = glfw.create_window(width, height, "Atom & Molecule Visualizer", None, None)
        glfw.make_context_current(self.win)

        imgui.create_context()
        self.impl = GlfwRenderer(self.win)

        self.trackball = Trackball()
        self.state = UIState()
        self.axes = Axes(
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main/gouraud.vert",
            "/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main/gouraud.frag",
            length=2.0
        ).setup()

        self.atom_model = None
        self.molecule_model = None

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0.05, 0.05, 0.1, 1.0)

        self.last_time = glfw.get_time()

    def run(self):
        while not glfw.window_should_close(self.win):
            current_time = glfw.get_time()
            delta_time = current_time - self.last_time
            self.last_time = current_time

            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()

            self.render_ui()
            self.update_scene(delta_time)

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            if self.state.show_axes:
                self.axes.draw(projection, view, np.eye(4))

            # Draw according to mode
            if self.state.mode_idx == 0 and self.atom_model:
                self.atom_model.draw(projection, view, np.eye(4))
            if self.state.mode_idx == 1 and self.molecule_model:
                self.molecule_model.draw(projection, view, np.eye(4))

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)

        self.impl.shutdown()
        glfw.terminate()

    def render_ui(self):
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(340, 260)
        imgui.begin("Controls", True)

        imgui.text("Mode:")
        _, self.state.mode_idx = imgui.combo("##mode", self.state.mode_idx, self.state.modes)
        imgui.separator()

        if self.state.mode_idx == 0:
            # Atom (Bohr)
            imgui.text("Atom (Bohr Model)")
            _, self.state.atom_idx = imgui.combo("##atom", self.state.atom_idx, self.state.atoms)
            current_atom = self.state.atoms[self.state.atom_idx]
            imgui.text(f"Current Atom: {current_atom}")
            _, self.state.animate_electron = imgui.checkbox("Animate Electrons", self.state.animate_electron)
        else:
            # Molecule (Ball-and-Stick)
            imgui.text("Molecule (Ball-and-Stick)")
            _, self.state.mol_idx = imgui.combo("##molecule", self.state.mol_idx, self.state.molecules)
            current_mol = self.state.molecules[self.state.mol_idx]
            imgui.text(f"Current Molecule: {current_mol}")
            _, self.state.animate_molecule = imgui.checkbox("Animate Molecule (optional)", self.state.animate_molecule)

        imgui.separator()
        _, self.state.show_axes = imgui.checkbox("Show Axes", self.state.show_axes)
        imgui.end()

    def update_scene(self, delta_time):
        if self.state.mode_idx == 0:
            # Atom mode
            atom_name = self.state.atoms[self.state.atom_idx]
            if self.atom_model is None or self.atom_model.atom_name != atom_name:
                self.atom_model = AtomModel(
                    atom_name=atom_name,
                    vert_shader="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main/gouraud.vert",
                    frag_shader="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main/gouraud.frag",
                    animate=self.state.animate_electron
                ).setup()
            self.atom_model.animate = self.state.animate_electron
            self.atom_model.update(delta_time)
            # release molecule model when not used (optional)
            self.molecule_model = None

        else:
            # Molecule mode
            mol_name = self.state.molecules[self.state.mol_idx]
            if self.molecule_model is None or self.molecule_model.molecule_name != mol_name:
                self.molecule_model = MoleculeModel(
                    molecule_name=mol_name,
                    # solid color shaders — see below
                    solid_vert="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/shaders/solid.vert",
                    solid_white="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/shaders/solid_white.frag",
                    solid_red="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/shaders/solid_red.frag",
                    solid_black="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/shaders/solid_black.frag",
                    solid_gray="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/atom/shaders/solid_gray.frag",
                    gouraud_vert="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main/gouraud.vert",
                    gouraud_frag="/Users/phamnguyenviettri/Ses251/ComputerGraphic/tostudents/main/gouraud.frag",
                    animate=self.state.animate_molecule
                ).setup()
            self.molecule_model.animate = self.state.animate_molecule
            self.molecule_model.update(delta_time)
            # release atom model when not used (optional)
            self.atom_model = None

    # input handlers...
    def on_key(self, win, key, scancode, action, mods):
        if action in (glfw.PRESS, glfw.REPEAT):
            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(self.win, True)

    def on_mouse_move(self, win, xpos, ypos):
        old = getattr(self, "mouse", (0, 0))
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, dx, dy):
        self.trackball.zoom(dy, glfw.get_window_size(win)[1])


def main():
    viewer = Viewer()
    viewer.run()


if __name__ == "__main__":
    main()
