#version 330 core
out vec4 fragColor;

uniform float time;  // ⏱️ nhận thời gian
uniform vec3 lightColor;
uniform vec3 lightDir;
in vec3 fragNormal;
in vec3 fragPos;

void main()
{
    vec3 norm = normalize(fragNormal);
    float diff = max(dot(norm, -lightDir), 0.0);

    float r = 0.5 + 0.5 * sin(time * 2.0);
    float g = 0.5 + 0.5 * sin(time * 2.0 + 2.0);
    float b = 0.5 + 0.5 * sin(time * 2.0 + 4.0);

    vec3 dynamicColor = vec3(r, g, b);
    vec3 result = dynamicColor * diff * lightColor;

    fragColor = vec4(result, 1.0);
}
