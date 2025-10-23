#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 projection;
uniform mat4 modelview;

out vec3 v_color;

void main() {
    // chỉ nội suy màu gốc giữa các đỉnh
    v_color = a_color;
    gl_Position = projection * modelview * vec4(a_position, 1.0);
}
