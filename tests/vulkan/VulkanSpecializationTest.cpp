#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ComputeBarycentric.h"
#include "VulkanCompute.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

class SpecializationProbe : public vulkan::VulkanCompute {
 public:
  SpecializationProbe(vulkan::VulkanContext* context,
                      vulkan::ComputeSpecialization specialization)
      : VulkanCompute(context), specialization_(std::move(specialization)) {}

  void Bind(const vulkan::VulkanBuffer& output) {
    CreateStorageBufferDescriptorSet(0, output);
    vkUpdateDescriptorSets(context_->logical_device,
                           static_cast<uint32_t>(writes_.size()),
                           writes_.data(), 0, nullptr);
  }

  void Run(VkCommandBuffer command) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set_, 0,
                            nullptr);
    vkCmdDispatch(command, 1, 1, 1);
  }

  mutable bool shader_loaded = false;

 private:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0,
             .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}};
  }

  vulkan::ComputeSpecialization GetSpecialization() const override {
    return specialization_;
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    shader_loaded = true;
    static const std::vector<uint32_t> code =
#include "ComputeSpecialization.comp.spv"
        ;
    return code;
  }

  vulkan::ComputeSpecialization specialization_;
};

void SubmitAndWait(vulkan::VulkanContext& context,
                   vulkan::VulkanCommandBuffer& command) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                       0, nullptr);
  vulkan::VulkanFence fence(&context);
  fence.Reset();
  command.Submit(fence.fence, vulkan::SubmitSyncInfo{});
  VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE,
                           UINT64_MAX));
}

TEST(VulkanSpecializationTest,
     UsesDefaultsAndTypedValuesWithIndependentPipelines) {
  vulkan::VulkanContext context(true);
  context.Init();
  vulkan::VulkanBuffer output(&context, 5 * sizeof(uint32_t),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vulkan::VulkanCommandBuffer command(&context);
  const uint32_t data[] = {8, 4, std::bit_cast<uint32_t>(int32_t{-19}),
                           std::bit_cast<uint32_t>(2.75F), VK_TRUE};
  vulkan::ComputeSpecialization specialization{
      .entries = {{.constantID = 0, .offset = 0, .size = 4},
                  {.constantID = 1, .offset = 4, .size = 4},
                  {.constantID = 7, .offset = 8, .size = 4},
                  {.constantID = 11, .offset = 12, .size = 4},
                  {.constantID = 13, .offset = 16, .size = sizeof(VkBool32)}},
      .data = std::vector<std::byte>(sizeof(data))};
  std::memcpy(specialization.data.data(), data, sizeof(data));
  SpecializationProbe specialized(&context, specialization);
  specialized.Init();
  specialized.Bind(output);
  specialization.data.clear();
  SpecializationProbe defaults(&context, {});
  defaults.Init();
  defaults.Bind(output);
  const uint32_t expected[][5] = {{data[2], data[3], 1, 8, 4},
                                  {std::bit_cast<uint32_t>(int32_t{-3}),
                                   std::bit_cast<uint32_t>(1.5F), 0, 1, 1}};
  for (int round = 0; round < 2; ++round) {
    for (int index = 0; index < 2; ++index) {
      output.MapData([](void* bytes) { std::memset(bytes, 0xff, 20); });
      command.BeginCommandBufferWithReset();
      (index == 0 ? specialized : defaults).Run(command.buffer());
      SubmitAndWait(context, command);
      output.MapData([&expected, index](void* bytes) {
        const uint32_t* actual = static_cast<const uint32_t*>(bytes);
        for (size_t element = 0; element < 5; ++element) {
          EXPECT_EQ(actual[element], expected[index][element]) << element;
        }
      });
    }
  }
}

TEST(VulkanSpecializationTest, RejectsMalformedMapsBeforeLoadingShader) {
  vulkan::VulkanContext context(true);
  context.Init();
  const std::vector<std::vector<VkSpecializationMapEntry>> invalid = {
      {{.constantID = 0, .offset = 0, .size = 0}},
      {{.constantID = 0, .offset = 4, .size = 1}},
      {{.constantID = 0, .offset = 1, .size = 4}},
      {{.constantID = 0, .offset = UINT32_MAX, .size = 4}},
      {{.constantID = 0, .offset = 0, .size = SIZE_MAX}},
      {{.constantID = 0, .offset = 0, .size = 4},
       {.constantID = 0, .offset = 0, .size = 4}}};
  for (const std::vector<VkSpecializationMapEntry>& entries : invalid) {
    SpecializationProbe probe(
        &context, {.entries = entries, .data = std::vector<std::byte>(4)});
    EXPECT_THROW(probe.Init(), std::invalid_argument);
    EXPECT_FALSE(probe.shader_loaded);
    EXPECT_EQ(probe.pipeline, VK_NULL_HANDLE);
  }
  SpecializationProbe empty(
      &context,
      {.entries = {{.constantID = 0, .offset = 0, .size = 4}}, .data = {}});
  EXPECT_THROW(empty.Init(), std::invalid_argument);
  EXPECT_FALSE(empty.shader_loaded);
}

TEST(VulkanSpecializationTest, RasterizesIdenticallyAcrossWorkgroupShapes) {
  vulkan::VulkanContext context(true);
  context.Init();
  using Rasterizer = example::ComputeBarycentric;
  const Rasterizer::WorkgroupSize groups[] = {{.x = 1, .y = 1},
                                              {.x = 8, .y = 4},
                                              {.x = 16, .y = 16},
                                              {.x = 32, .y = 1}};
  const glm::ivec2 shapes[] = {glm::ivec2(1, 1), glm::ivec2(37, 29),
                               glm::ivec2(32, 16)};
  vulkan::VulkanCommandBuffer command(&context);
  for (const glm::ivec2& shape : shapes) {
    for (const Rasterizer::WorkgroupSize& group : groups) {
      Rasterizer rasterizer(&context, shape.x, shape.y, group);
      rasterizer.Init();
      const Rasterizer::Triangle triangle{.a = glm::ivec2(-1, -1),
                                          .b = glm::ivec2(shape.x - 1, -1),
                                          .c = glm::ivec2(-1, shape.y - 1)};
      for (int round = 0; round < 2; ++round) {
        rasterizer.rasterized.MapData([&rasterizer](void* bytes) {
          std::memset(bytes, 0,
                      static_cast<size_t>(rasterizer.rasterized.Size()));
        });
        command.BeginCommandBufferWithReset();
        rasterizer.Run(command.buffer(), triangle);
        SubmitAndWait(context, command);
        rasterizer.rasterized.MapData([&shape](void* bytes) {
          const int* pixels = static_cast<const int*>(bytes);
          for (int y = 0; y < shape.y; ++y) {
            for (int x = 0; x < shape.x; ++x) {
              const bool covered =
                  shape.x * shape.y - shape.x * (y + 1) - shape.y * (x + 1) >=
                  0;
              ASSERT_EQ(pixels[y * shape.x + x], covered ? 1 : 0)
                  << shape.x << "," << shape.y << " at " << x << "," << y;
            }
          }
        });
      }
    }
  }
}

TEST(VulkanSpecializationTest, RejectsUnsupportedWorkgroupsBeforeAllocating) {
  vulkan::VulkanContext context(true);
  context.Init();
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  const VkPhysicalDeviceLimits& limits = properties.limits;
  using Rasterizer = example::ComputeBarycentric;
  const Rasterizer::WorkgroupSize groups[] = {
      {.x = 0, .y = 1},
      {.x = 1, .y = 0},
      {.x = UINT32_MAX, .y = 1},
      {.x = 1, .y = UINT32_MAX},
      {.x = limits.maxComputeWorkGroupSize[0],
       .y = limits.maxComputeWorkGroupSize[1]}};
  for (const Rasterizer::WorkgroupSize& group : groups) {
    if (static_cast<uint64_t>(group.x) * group.y >
            limits.maxComputeWorkGroupInvocations ||
        group.x == 0 || group.y == 0 ||
        group.x > limits.maxComputeWorkGroupSize[0] ||
        group.y > limits.maxComputeWorkGroupSize[1]) {
      EXPECT_THROW((Rasterizer(&context, 37, 29, group)),
                   std::invalid_argument);
    }
  }
  Rasterizer valid(&context, 37, 29, {.x = 8, .y = 4});
  valid.Init();
  EXPECT_NE(valid.pipeline, VK_NULL_HANDLE);
}

}
}
}
