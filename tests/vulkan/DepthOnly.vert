#version 450

layout(push_constant) uniform DepthRamp {
    float left;
    float right;
} ramp;

void main() {
    const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    const vec2 position = positions[gl_VertexIndex];
    const float depth = mix(ramp.left, ramp.right, (position.x + 1.0) * 0.5);
    gl_Position = vec4(position, depth, 1.0);
}
