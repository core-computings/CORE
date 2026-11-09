#version 450

layout(binding = 0) uniform Color {
    vec4 value;
} color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = color.value;
}
