#version 450

layout(push_constant) uniform Draw {
  vec4 color;
  float depth;
} draw;

void main() {
  const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  gl_Position = vec4(positions[gl_VertexIndex], draw.depth, 1.0);
}
