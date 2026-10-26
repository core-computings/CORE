struct UniformData {
    int width;
    int height;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformData> ubo;

[[vk::binding(1, 0)]]
StructuredBuffer<float> src;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> dst;

[[vk::binding(3, 0)]]
StructuredBuffer<float> kernel;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    const int row = int(dispatchThreadID.y);
    const int col = int(dispatchThreadID.x);
    if (row >= ubo.height || col >= ubo.width) {
        return;
    }

    const int origIdx = row * ubo.width + col;
    if (row == 0 || row == (ubo.height - 1) || col == 0 || col == (ubo.width - 1)) {
        dst[origIdx] = src[origIdx];
        return;
    }

    float sum = 0.0f;
    int idx = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            const int newIdx = (row + i) * ubo.width + (col + j);
            sum += src[newIdx] * kernel[idx];
            ++idx;
        }
    }

    dst[origIdx] = sum;
}
