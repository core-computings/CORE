#version 450

layout(push_constant) uniform Draw {
  vec4 color;
  float depth;
} draw;

layout(location = 0) out vec4 out_color;

void main() { out_color = draw.color; }
