#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ComputeFma.h"
#include "VulkanCommandBuffer.h"
#include "VulkanCompute.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

constexpr uint32_t kLocalSize = 64;
constexpr uint32_t kGuardHalves = 8;
constexpr uint16_t kGuardBits = 0xffff;

uint16_t ExactHalfBits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  if ((bits & 0x7fffffffu) == 0) {
    return static_cast<uint16_t>(sign);
  }
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  const uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0 || exponent >= 31 || (mantissa & 0x1fffu) != 0) {
    throw std::invalid_argument("Value is not an exact normal half");
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

class Float16Probe : public vulkan::VulkanCompute {
 public:
  explicit Float16Probe(vulkan::VulkanContext* context) : VulkanCompute(context) {}

  void Bind(const vulkan::VulkanBuffer& source, const vulkan::VulkanBuffer& destination) {
    CreateStorageBufferDescriptorSet(0, source);
    CreateStorageBufferDescriptorSet(1, destination);
    vkUpdateDescriptorSets(context_->logical_device, static_cast<uint32_t>(writes_.size()),
                           writes_.data(), 0, nullptr);
  }

  void Run(VkCommandBuffer command, uint32_t count) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                            &descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count),
                       &count);
    vkCmdDispatch(command, (count + kLocalSize - 1) / kLocalSize, 1, 1);
  }

 private:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0,
             .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1,
             .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}};
  }

  std::vector<VkPushConstantRange> GetPushConstantRanges() const override {
    return {{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(uint32_t)}};
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> code =
#include "Float16Storage.comp.spv"
        ;
    return code;
  }
};

TEST(VulkanFloat16Test, ComputesExactHalfResultsThroughHalfStorage) {
  vulkan::VulkanContext context(true);
  context.Init();
  const vulkan::Float16Support& support = context.float16_support();
  if (!support.arithmetic || !support.storage_buffer) {
    GTEST_SKIP() << "Float16 arithmetic or 16-bit storage buffers are not supported";
  }
  constexpr VkMemoryPropertyFlags kHost =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (const uint32_t count : {1u, 63u, 64u, 1000u}) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count + kGuardHalves) * sizeof(uint16_t);
    vulkan::VulkanBuffer source(&context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHost);
    vulkan::VulkanBuffer destination(&context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHost);
    std::vector<uint16_t> input(count + kGuardHalves, kGuardBits);
    std::vector<uint16_t> expected(count + kGuardHalves, kGuardBits);
    for (uint32_t index = 0; index < count; ++index) {
      const float value = static_cast<float>(static_cast<int32_t>(index % 63u) - 31);
      input[index] = ExactHalfBits(value);
      expected[index] = ExactHalfBits(value * value * 0.125f - value * 0.5f + 0.75f);
    }
    source.MapData([&input](void* data) {
      std::memcpy(data, input.data(), input.size() * sizeof(uint16_t));
    });
    destination.MapData([count](void* data) {
      std::memset(data, 0xff, (count + kGuardHalves) * sizeof(uint16_t));
    });
    Float16Probe probe(&context);
    probe.Init();
    probe.Bind(source, destination);
    vulkan::VulkanCommandBuffer command(&context);
    command.BeginCommandBufferWithReset();
    probe.Run(command.buffer(), count);
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vulkan::VulkanFence fence(&context);
    fence.Reset();
    command.Submit(fence.fence, vulkan::SubmitSyncInfo{});
    VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
    destination.MapData([&expected, count](void* data) {
      const uint16_t* actual = static_cast<const uint16_t*>(data);
      for (uint32_t index = 0; index < count + kGuardHalves; ++index) {
        ASSERT_EQ(actual[index], expected[index]) << "count " << count << " index " << index;
      }
    });
    source.MapData([&input](void* data) {
      EXPECT_EQ(std::memcmp(data, input.data(), input.size() * sizeof(uint16_t)), 0);
    });
  }
}

TEST(VulkanFloat16Test, RooflineFloat16KernelRequiresArithmeticSupport) {
  vulkan::VulkanContext context(true);
  context.Init();
  vulkan::VulkanBuffer output(&context, vulkan::ComputeFma::kLocalSizeX * sizeof(float),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (!context.float16_support().arithmetic) {
    EXPECT_THROW(vulkan::ComputeFma(&context, output, 1, 1, vulkan::FmaPrecision::kFloat16),
                 std::invalid_argument);
    return;
  }
  vulkan::ComputeFma fma(&context, output, 64, 1, vulkan::FmaPrecision::kFloat16);
  EXPECT_EQ(fma.FlopPerIteration(), vulkan::ComputeFma::kFloat16FlopPerIteration);
  fma.Init();
  vulkan::VulkanCommandBuffer command(&context);
  command.BeginCommandBufferWithReset();
  fma.Run(command.buffer());
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
  };
  vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  vulkan::VulkanFence fence(&context);
  fence.Reset();
  command.Submit(fence.fence, vulkan::SubmitSyncInfo{});
  VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
  output.MapData([](void* data) {
    const float* values = static_cast<const float*>(data);
    for (uint32_t index = 0; index < vulkan::ComputeFma::kLocalSizeX; ++index) {
      ASSERT_NEAR(values[index], 4.0f, 0.01f) << index;
    }
  });
}

}
}
}
