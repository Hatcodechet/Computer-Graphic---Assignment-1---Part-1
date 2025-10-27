#version 330 core
in vec3 v_position;  // World space position
in vec3 v_normal;    // World space normal
in vec3 v_color;

out vec4 fragColor;

// Light and view positions in world space
uniform vec3 light_pos = vec3(3.0, 3.0, 3.0);
uniform vec3 view_pos = vec3(0.0, 0.0, 5.0);
uniform float ka = 0.1;
uniform float kd = 0.8;
uniform float ks = 0.5;
uniform float shininess = 32.0;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_pos - v_position);
    vec3 V = normalize(view_pos - v_position);
    vec3 R = reflect(-L, N);

    vec3 ambient = ka * v_color;
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = kd * diff * v_color;
    float spec = pow(max(dot(R, V), 0.0), shininess);
    vec3 specular = ks * spec * vec3(1.0);

    vec3 color = ambient + diffuse + specular;
    fragColor = vec4(color, 1.0);
}
