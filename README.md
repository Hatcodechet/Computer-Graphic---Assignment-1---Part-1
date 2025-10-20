# 🚀 Computer Graphics with Modern OpenGL (Python + PyOpenGL)

This project contains **basic OpenGL samples** using **Modern OpenGL**, organized for students in a Computer Graphics course. It includes reusable utility libraries and several illustrative examples of rendering objects using shaders, buffers, and transformation matrices.

---

## 📁 Project Structure

```bash
tostudents/
│
├── libs/                  # Shared helper modules
│   ├── buffer.py          # Manages VAO, VBO, and EBO
│   ├── shader.py          # Shader compilation & linking
│   └── transform.py       # Transformation utilities (translate, rotate, scale)
│
├── triangle_basic/        # Example 1: Hello Triangle
│   ├── view.py            # Entry point to run the OpenGL app
│   ├── triangle.vert      # Vertex shader
│   └── triangle.frag      # Fragment shader
│
├── cube_rotate/           # Example 2: Rotating Cube
│   ├── view.py
│   ├── cube.vert
│   └── cube.frag
│
├── cube_texture/          # Example 3: Textured Cube
│   ├── view.py
│   ├── texture.png        # Texture image used for cube
│   ├── cube.vert
│   └── cube.frag
│
└── README.md              # 📄 You are here!
```

---

## 🛠️ Requirements

You need Python **≥ 3.8** and the following packages:

```bash
pip install PyOpenGL PyOpenGL_accelerate opencv-python glfw numpy
```

### ✔️ Tested with:
- Python 3.10
- GLFW 3.3
- PyOpenGL 3.1.6
- OpenCV 4.x

---

## 🧪 How to Run Examples

Each sample folder contains a `view.py` file. To run an example:

### 🔺 Run Hello Triangle

```bash
cd triangle_basic
python view.py
```

### 🧊 Run Rotating Cube

```bash
cd cube_rotate
python view.py
```

### 🎨 Run Textured Cube

```bash
cd cube_texture
python view.py
```

Make sure your environment supports OpenGL and you are not in headless mode (you need a display).

---

## 🔍 File Responsibilities

| File              | Description |
|-------------------|-------------|
| `libs/buffer.py`  | Defines reusable classes and functions to create VAO, VBO, EBO for objects. |
| `libs/shader.py`  | Loads and compiles shaders (GLSL). Also handles program linking. |
| `libs/transform.py` | Offers matrix utilities for 3D transformations. |
| `view.py` (each sample) | Initializes window and OpenGL context. Loads shaders, buffers, handles rendering loop. |
| `.vert` files     | Vertex shader code in GLSL. |
| `.frag` files     | Fragment shader code in GLSL. |

---

## 🧠 Learning Objectives

- Understand how OpenGL buffers (VAO, VBO) are created and managed.
- Learn GLSL shader programming (vertex + fragment).
- Apply transformations (rotate, scale, translate) via matrices.
- Load textures and map them onto 3D geometry.
- Separate rendering logic and reusability using modular Python code.

---

## 📦 Optional: Run in Google Colab (with GUI support)

If running in Google Colab (with virtual display):

```python
!apt install -y xvfb
!pip install PyOpenGL glfw opencv-python pyvirtualdisplay

from pyvirtualdisplay import Display
Display(visible=0, size=(800,600)).start()

!python view.py  # assuming you upload and set up the code
```

---

## 📘 Suggested Extensions

- Add keyboard or mouse interaction to rotate/view objects.
- Load 3D models from `.obj` files using `assimp` or custom loader.
- Animate transformations with time delta.
- Try perspective vs orthographic projection matrix.

---

## 🙋‍♂️ Troubleshooting

- **Black screen?**
  - Check shader compilation log.
  - Make sure VAO is activated before `glDraw*`.

- **GLFW Error?**
  - Ensure your system supports OpenGL 3.3+.
  - Try updating GPU drivers or use CPU OpenGL emulation (`Mesa`).

- **Texture not loading?**
  - Make sure `texture.png` is present and `cv2.imread()` works.

---

## 👨‍🏫 Instructor Notes (optional)

- Each sample represents a minimal unit of concept: triangle, cube, texture.
- Students can extend them by adding animation, lighting, or camera controls.
- Designed to be used with slides from [learnopengl.com](https://learnopengl.com/).
