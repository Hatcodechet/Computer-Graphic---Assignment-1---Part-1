#version 330 core
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_color;
layout(location=2) in vec3 a_normal;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;

uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;

void main() {
    mat4 modelview = view * model;
    gl_Position = projection * modelview * vec4(a_position, 1.0);
    
    // Transform position to world space for lighting calculations
    vec4 world_pos = model * vec4(a_position, 1.0);
    v_position = vec3(world_pos);
    
    // Transform normal to world space using normal matrix
    mat3 normal_matrix = mat3(transpose(inverse(model)));
    v_normal = normalize(normal_matrix * a_normal);
    
    v_color = a_color;
}
