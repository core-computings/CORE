#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ComputeGaussianBlur.h"
#include "GaussianBlurBaseline.h"
#include "GaussianBlurReference.h"
#include "VulkanCommandBuffer.h"
#include "VulkanSync.h"

namespace {

using core::vulkan::ComputeGaussianBlur;
using core::vulkan::VulkanBuffer;
using core::vulkan::VulkanContext;

class DispatchTimer {
 public:
  explicit DispatchTimer(VulkanContext* context) : context_(context) {
    const VkQueryPoolCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
        .pipelineStatistics = 0,
    };
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device, &count, families.data());
    const uint32_t bits =
        families[context_->GetQueueFamilyIndices().compute_family.value()].timestampValidBits;
    if (bits == 0) {
      throw std::runtime_error("Compute queue does not support timestamps");
    }
    mask_ = UINT64_MAX >> (64U - bits);
    VK_CHECK(vkCreateQueryPool(context_->logical_device, &info, nullptr, &pool_));
  }

  ~DispatchTimer() { vkDestroyQueryPool(context_->logical_device, pool_, nullptr); }

  template <typename Pipeline>
  double Run(Pipeline& blur, core::vulkan::VulkanCommandBuffer& command,
             core::vulkan::VulkanFence& fence) {
    fence.Reset();
    command.BeginCommandBufferWithReset();
    vkCmdResetQueryPool(command.buffer(), pool_, 0, 2);
    vkCmdWriteTimestamp(command.buffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool_, 0);
    blur.Run(command.buffer());
    vkCmdWriteTimestamp(command.buffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, 1);
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context_->logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
    std::array<uint64_t, 2> timestamps{};
    VK_CHECK(vkGetQueryPoolResults(context_->logical_device, pool_, 0, 2, sizeof(timestamps),
                                   timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
    const double microseconds = static_cast<double>((timestamps[1] - timestamps[0]) & mask_) *
                                static_cast<double>(context_->timestamp_period) / 1000.0;
    if (microseconds <= 0.0) {
      throw std::runtime_error("Invalid GPU timestamp interval");
    }
    return microseconds;
  }

 private:
  VulkanContext* context_;
  VkQueryPool pool_ = VK_NULL_HANDLE;
  uint64_t mask_ = UINT64_MAX;
};

void Check(VulkanBuffer& buffer, const std::vector<float>& expected) {
  buffer.MapData([&expected](void* data) {
    const float* actual = static_cast<const float*>(data);
    if (!std::equal(expected.begin(), expected.end(), actual)) {
      throw std::runtime_error("Blur differs from the CPU oracle");
    }
  });
}

void Measure(VulkanContext& context, int width, int height) {
  const std::vector<float> input = core::test::GaussianInput(width, height);
  const std::vector<float> expected = core::test::GaussianReference(input, width, height);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(input.size()) * sizeof(float);
  constexpr VkMemoryPropertyFlags kHostMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VulkanBuffer source(&context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
  VulkanBuffer destination(&context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
  source.MapData([&input, bytes](void* data) { std::memcpy(data, input.data(), bytes); });
  core::test::BaselineBlur baseline(&context, source, destination, width, height);
  ComputeGaussianBlur candidate(&context, source, destination, width, height);
  baseline.Init();
  candidate.Init();
  core::vulkan::VulkanCommandBuffer command(&context);
  core::vulkan::VulkanFence fence(&context);
  DispatchTimer timer(&context);
  std::array<std::vector<double>, 2> samples;
  constexpr int kWarmups = 5;
  constexpr int kSamples = 31;
  for (int run = 0; run < kWarmups + kSamples; ++run) {
    for (int turn = 0; turn < 2; ++turn) {
      const size_t variant = static_cast<size_t>((run + turn) % 2);
      const double elapsed = variant == 0 ? timer.Run(baseline, command, fence)
                                         : timer.Run(candidate, command, fence);
      if (run >= kWarmups) {
        samples[variant].push_back(elapsed);
        std::printf("sample,%d,%d,%zu,%d,%.3f\n", width, height, variant, run - kWarmups,
                    elapsed);
      }
      if (run == 0 || run == kWarmups + kSamples - 1) {
        Check(destination, expected);
      }
    }
  }
  Check(source, input);
  for (size_t variant = 0; variant < samples.size(); ++variant) {
    std::vector<double>& times = samples[variant];
    std::sort(times.begin(), times.end());
    std::printf("summary,%d,%d,%zu,%d,%.3f,%.3f,%.3f,verified\n", width, height, variant,
                kSamples, times[3], times[15], times[27]);
  }
}

}

int main() {
  VulkanContext context(false, core::vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  std::printf("device,%s,driver,%u,timestamp_ns,%.3f\n", properties.deviceName,
              properties.driverVersion, static_cast<double>(properties.limits.timestampPeriod));
  Measure(context, 64, 64);
  Measure(context, 1024, 1024);
  Measure(context, 1003, 769);
  Measure(context, 4000, 3000);
}
