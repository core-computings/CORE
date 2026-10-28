#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanContext.h"

namespace {

void Benchmark(core::vulkan::VulkanContext& context, size_t count, size_t bytes,
               size_t iterations) {
  constexpr VkMemoryPropertyFlags kHostMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  std::array<std::vector<std::unique_ptr<core::vulkan::VulkanBuffer>>, 2> buffers;
  for (std::vector<std::unique_ptr<core::vulkan::VulkanBuffer>>& group : buffers) {
    for (size_t index = 0; index < count; ++index) {
      group.push_back(std::make_unique<core::vulkan::VulkanBuffer>(
          &context, bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          kHostMemory));
    }
  }
  core::vulkan::VulkanBuffer readback(
      &context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostMemory);
  std::vector<uint8_t> input(bytes);
  std::array<std::vector<double>, 2> samples;
  constexpr int kWarmups = 5;
  constexpr int kSamples = 31;
  for (int sample = -kWarmups; sample < kSamples; ++sample) {
    for (size_t byte = 0; byte < bytes; ++byte) {
      input[byte] = static_cast<uint8_t>((byte * 17 + static_cast<size_t>(sample + kWarmups)) % 251);
    }
    for (size_t turn = 0; turn < 2; ++turn) {
      const size_t mode = (static_cast<size_t>(sample + kWarmups) + turn) % 2;
      const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (const std::unique_ptr<core::vulkan::VulkanBuffer>& buffer : buffers[mode]) {
          const auto update = [&input, bytes](void* data) {
            std::memcpy(data, input.data(), bytes);
          };
          if (mode == 0) {
            buffer->MapData(update);
          } else {
            buffer->MapDataPersistent(update);
          }
        }
      }
      const double elapsed =
          std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
      if (sample >= 0) {
        samples[mode].push_back(elapsed / static_cast<double>(iterations * count));
      }
      for (const std::unique_ptr<core::vulkan::VulkanBuffer>& buffer : buffers[mode]) {
        buffer->CopyToBuffer(readback);
        readback.MapData([&input, bytes](void* data) {
          if (std::memcmp(data, input.data(), bytes) != 0) {
            throw std::runtime_error("Mapped update GPU readback mismatch");
          }
        });
      }
    }
  }
  for (std::vector<double>& mode_samples : samples) {
    std::sort(mode_samples.begin(), mode_samples.end());
  }
  std::printf("buffers=%zu bytes=%zu updates_per_buffer=%zu warmups=%d samples=%d "
              "transient_ns[p10,median,p90]=[%.2f,%.2f,%.2f] "
              "persistent_ns[p10,median,p90]=[%.2f,%.2f,%.2f] verified=yes\n",
              count, bytes, iterations, kWarmups, kSamples, samples[0][3], samples[0][15],
              samples[0][27], samples[1][3], samples[1][15], samples[1][27]);
}

}

int main() {
  core::vulkan::VulkanContext context(
      false, core::vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  std::printf("GPU=%s; validation disabled; CPU wall time includes callback and memcpy; "
              "allocation/input generation/GPU copy/readback excluded; warmed updates only\n",
              properties.deviceName);
  Benchmark(context, 1, 192, 10000);
  Benchmark(context, 64, 192, 1000);
  Benchmark(context, 1, 4096, 10000);
  Benchmark(context, 1, 1048576, 32);
  return 0;
}
