#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ComputeCopy.h"
#include "VulkanCompute.h"
#include "VulkanSampler.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

static_assert(!std::is_copy_constructible_v<vulkan::VulkanBase>);
static_assert(!std::is_copy_assignable_v<vulkan::VulkanBase>);
static_assert(!std::is_copy_constructible_v<vulkan::ComputeCopy>);
static_assert(!std::is_copy_assignable_v<vulkan::ComputeCopy>);

constexpr VkMemoryPropertyFlags kHostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

void SubmitAndWait(vulkan::VulkanContext& context,
                   vulkan::VulkanCommandBuffer& command,
                   vulkan::VulkanFence& fence) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                       0, nullptr);
  fence.Reset();
  command.Submit(fence.fence, vulkan::SubmitSyncInfo{});
  VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE,
                           UINT64_MAX));
}

class DescriptorProbe : public vulkan::VulkanCompute {
 public:
  DescriptorProbe(vulkan::VulkanContext* context,
                  std::vector<vulkan::BindingInfo> bindings)
      : VulkanCompute(context), bindings_(std::move(bindings)) {}

  using VulkanBase::CreateCombinedImageSamplerDescriptorSet;
  using VulkanBase::CreateStorageBufferDescriptorSet;
  using VulkanBase::CreateUniformBufferDescriptorSet;

  const std::vector<VkWriteDescriptorSet>& writes() const { return writes_; }

  void Update() {
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

 private:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return bindings_;
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> code =
#include "DescriptorRebinding.comp.spv"
        ;
    return code;
  }

  std::vector<vulkan::BindingInfo> bindings_;
};

TEST(VulkanDescriptorRebindingTest,
     ReusesComputeCopyAcrossCompletedBufferPairs) {
  vulkan::VulkanContext context(true);
  context.Init();
  constexpr int kCount = 513;
  constexpr size_t kWords = static_cast<size_t>(kCount) * 4;
  constexpr size_t kGuard = 7;
  constexpr VkDeviceSize kBytes = (kWords + kGuard) * sizeof(uint32_t);
  std::array<vulkan::VulkanBuffer, 3> buffers;
  for (vulkan::VulkanBuffer& buffer : buffers) {
    buffer = vulkan::VulkanBuffer(
        &context, kBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
  }
  vulkan::ComputeCopy copy(&context, buffers[0], buffers[1], kCount);
  copy.Init();
  const VkPipeline pipeline = copy.pipeline;
  const VkPipelineLayout layout = copy.pipeline_layout;
  vulkan::VulkanCommandBuffer command(&context);
  vulkan::VulkanFence fence(&context);
  vulkan::VulkanBuffer short_buffer(
      &context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
  std::vector<uint32_t> input(kWords + kGuard);
  for (size_t round = 0; round < 6; ++round) {
    SCOPED_TRACE(round);
    vulkan::VulkanBuffer& source = buffers[round % buffers.size()];
    vulkan::VulkanBuffer& destination = buffers[(round + 1) % buffers.size()];
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] =
          std::bit_cast<uint32_t>(static_cast<float>(round * 10000 + index));
    }
    source.MapData(
        [&input](void* data) { std::memcpy(data, input.data(), kBytes); });
    destination.MapData([](void* data) { std::memset(data, 0xff, kBytes); });
    command.Reset();
    if (round > 0) {
      copy.BindBuffers(source, destination);
    }
    if (round == 5) {
      EXPECT_THROW(copy.BindBuffers(short_buffer, destination),
                   std::invalid_argument);
      EXPECT_THROW(copy.BindBuffers(source, short_buffer),
                   std::invalid_argument);
      EXPECT_THROW(copy.BindBuffers(source, source), std::invalid_argument);
    }
    command.BeginCommandBuffer();
    copy.Run(command.buffer());
    SubmitAndWait(context, command, fence);
    destination.MapData([&input](void* data) {
      const uint32_t* actual = static_cast<const uint32_t*>(data);
      for (size_t index = 0; index < input.size(); ++index) {
        ASSERT_EQ(actual[index], index < kWords ? input[index] : UINT32_MAX)
            << index;
      }
    });
    source.MapData([&input](void* data) {
      EXPECT_EQ(std::memcmp(data, input.data(), kBytes), 0);
    });
    EXPECT_EQ(copy.pipeline, pipeline);
    EXPECT_EQ(copy.pipeline_layout, layout);
  }
}

TEST(VulkanDescriptorRebindingTest, ReplacesSparseMixedBindingsInAnyOrder) {
  vulkan::VulkanContext context(true);
  context.Init();
  DescriptorProbe probe(
      &context, {{.binding = 21,
                  .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
                 {.binding = 7,
                  .descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                  .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
                 {.binding = 3,
                  .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
                 {.binding = 11,
                  .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}});
  std::array<vulkan::VulkanBuffer, 2> inputs;
  std::array<vulkan::VulkanBuffer, 2> uniforms;
  std::array<vulkan::VulkanBuffer, 2> outputs;
  std::array<vulkan::VulkanImage, 2> images;
  vulkan::VulkanSampler sampler(&context);
  for (size_t index = 0; index < inputs.size(); ++index) {
    inputs[index] = vulkan::VulkanBuffer(
        &context, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
    uniforms[index] = vulkan::VulkanBuffer(
        &context, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostMemory);
    outputs[index] = vulkan::VulkanBuffer(
        &context, 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
    inputs[index].MapData([index](void* data) {
      uint32_t* values = static_cast<uint32_t*>(data);
      for (uint32_t element = 0; element < 8; ++element) {
        values[element] = static_cast<uint32_t>(index) * 100 + element;
      }
    });
    uniforms[index].MapData([index](void* data) {
      const uint32_t addend = static_cast<uint32_t>(index) * 1000 + 17;
      std::memcpy(data, &addend, sizeof(addend));
    });
    images[index] = vulkan::VulkanImage(
        &context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vulkan::VulkanBuffer staging(&context, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 kHostMemory);
    staging.MapData([index](void* data) {
      const std::array<uint8_t, 4> pixel = {
          static_cast<uint8_t>(index * 100 + 31), 0, 0, 255};
      std::memcpy(data, pixel.data(), pixel.size());
    });
    vulkan::VulkanCommandBuffer upload =
        vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
    images[index].TransitionImageLayout(
        upload.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_FORMAT_R8G8B8A8_UNORM);
    staging.CopyToImage(upload.buffer(), images[index], 1, 1);
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = images[index].image;
    barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
    vkCmdPipelineBarrier(upload.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    vulkan::VulkanFence fence(&context);
    fence.Reset();
    upload.Submit(fence.fence, vulkan::SubmitSyncInfo{});
    VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE,
                             UINT64_MAX));
  }
  EXPECT_THROW(probe.CreateStorageBufferDescriptorSet(11, inputs[0]),
               std::logic_error);
  probe.Init();
  const VkPipeline pipeline = probe.pipeline;
  const VkWriteDescriptorSet* writes = probe.writes().data();
  vulkan::VulkanCommandBuffer command(&context);
  vulkan::VulkanFence fence(&context);
  for (size_t round = 0; round < 4; ++round) {
    SCOPED_TRACE(round);
    command.Reset();
    for (vulkan::VulkanBuffer& output : outputs) {
      output.MapData([](void* data) { std::memset(data, 0xff, 48); });
    }
    for (size_t replacement = 0; replacement < 20; ++replacement) {
      const size_t index = (replacement + round) % 2;
      probe.CreateCombinedImageSamplerDescriptorSet(3, images[index].image_view,
                                                    sampler.sampler);
      probe.CreateStorageBufferDescriptorSet(11, inputs[index]);
      probe.CreateUniformBufferDescriptorSet(7, uniforms[index]);
      probe.CreateStorageBufferDescriptorSet(21, outputs[index]);
    }
    EXPECT_THROW(probe.CreateStorageBufferDescriptorSet(7, inputs[0]),
                 std::invalid_argument);
    EXPECT_THROW(probe.CreateUniformBufferDescriptorSet(11, uniforms[0]),
                 std::invalid_argument);
    EXPECT_THROW(probe.CreateCombinedImageSamplerDescriptorSet(
                     21, images[0].image_view, sampler.sampler),
                 std::invalid_argument);
    EXPECT_THROW(probe.CreateStorageBufferDescriptorSet(0, inputs[0]),
                 std::invalid_argument);
    ASSERT_EQ(probe.writes().size(), 4U);
    EXPECT_EQ(probe.writes().data(), writes);
    probe.Update();
    command.BeginCommandBuffer();
    probe.Run(command.buffer());
    SubmitAndWait(context, command, fence);
    const size_t selected = (19 + round) % 2;
    for (size_t index = 0; index < outputs.size(); ++index) {
      outputs[index].MapData([index, selected](void* data) {
        const uint32_t* values = static_cast<const uint32_t*>(data);
        for (uint32_t element = 0; element < 12; ++element) {
          const uint32_t expected =
              index == selected && element < 8
                  ? static_cast<uint32_t>(index) * 1200 + element + 48
                  : UINT32_MAX;
          ASSERT_EQ(values[element], expected) << index << ":" << element;
        }
      });
    }
    EXPECT_EQ(probe.pipeline, pipeline);
  }
}

TEST(VulkanDescriptorRebindingTest,
     RejectsDuplicateLayoutBindingsBeforeAllocation) {
  vulkan::VulkanContext context(true);
  context.Init();
  DescriptorProbe probe(&context,
                        {{.binding = 11,
                          .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
                         {.binding = 11,
                          .descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}});
  EXPECT_THROW(probe.Init(), std::invalid_argument);
  EXPECT_EQ(probe.pipeline, VK_NULL_HANDLE);
  EXPECT_EQ(probe.pipeline_layout, VK_NULL_HANDLE);
  EXPECT_TRUE(probe.writes().empty());
}

}
}
}
