#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec2 fragWorldXZ;
layout(location = 2) in vec4 fragLightClip;

layout(binding = 1) uniform sampler2D groundTex;
layout(binding = 2) uniform sampler2DShadow shadowMap;

layout(location = 0) out vec4 outColor;

float ShadowVisibility() {
    vec3 light = fragLightClip.xyz / fragLightClip.w;
    vec2 uv = light.xy * 0.5 + 0.5;
    if (light.z <= 0.0 || light.z >= 1.0) {
        return 1.0;
    }
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            visibility += texture(shadowMap, vec3(uv + vec2(x, y) * texel, light.z));
        }
    }
    return visibility / 9.0;
}

void main() {
    vec3 sampled = texture(groundTex, fragUV).rgb;
    sampled *= mix(0.45, 1.0, ShadowVisibility());

    // Fade the textured plane toward the horizon color so its edge doesn't
    // read as a hard seam against the skybox.
    float dist = length(fragWorldXZ);
    float fade = clamp(1.0 - dist / 70.0, 0.0, 1.0);
    vec3 horizon = vec3(0.58, 0.62, 0.66);
    outColor = vec4(mix(horizon, sampled, fade), 1.0);
}
