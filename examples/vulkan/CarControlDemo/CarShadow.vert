#version 450

layout(push_constant) uniform ShadowTransform {
    mat4 light_model_view_projection;
} transform;

layout(location = 0) in vec3 inPos;

void main() {
    gl_Position = transform.light_model_view_projection * vec4(inPos, 1.0);
}
