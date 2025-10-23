#version 330 core
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_color;
layout(location=2) in vec3 a_normal;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;

uniform mat4 projection;
uniform mat4 modelview;

void main() {
    gl_Position = projection * modelview * vec4(a_position, 1.0);
    v_position = vec3(modelview * vec4(a_position, 1.0));
    v_normal = normalize(mat3(transpose(inverse(modelview))) * a_normal);
    v_color = a_color;
}
