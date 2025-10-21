#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 vertexColor;

out vec3 vColor;
out vec3 vNormal;
out vec3 vPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    vPos = worldPos.xyz;
    vColor = vertexColor;
    vNormal = normalize(mat3(model) * vec3(0.0, 0.0, 1.0)); // giả lập normal 2D hướng ra z
    gl_Position = projection * view * worldPos;
}
