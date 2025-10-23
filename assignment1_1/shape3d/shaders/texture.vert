#version 330 core
layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;
layout(location = 2) in vec3 vertex_color;
layout(location = 3) in vec2 vertex_texcoord;

uniform mat4 projection;
uniform mat4 modelview;

out vec2 TexCoord;

void main() {
    gl_Position = projection * modelview * vec4(vertex_position, 1.0);
    TexCoord = vertex_texcoord;
}
