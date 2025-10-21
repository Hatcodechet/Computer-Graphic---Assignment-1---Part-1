#version 330 core
in vec3 vColor;
in vec3 vNormal;
in vec3 vPos;

out vec4 fragColor;

uniform vec3 lightPos = vec3(1.0, 1.0, 1.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 viewPos = vec3(0.0, 0.0, 2.0);

void main() {
    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(lightPos - vPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * vColor;

    vec3 ambient = 0.2 * vColor;
    vec3 result = ambient + diffuse;

    fragColor = vec4(result, 1.0);
}
