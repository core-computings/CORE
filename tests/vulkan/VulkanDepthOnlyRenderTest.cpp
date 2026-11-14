#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanRender.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

constexpr uint32_t kSize = 32;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

struct DepthRamp {
  float left;
  float right;
};

std::vector<uint32_t> DepthOnlyVertexShader() {
  static const std::vector<uint32_t> code =
#include "DepthOnly.vert.spv"
      ;
  return code;
}

class DepthOnlyPipeline : public vulkan::VulkanRender {
 public:
  DepthOnlyPipeline(vulkan::VulkanContext* context, const vulkan::DynamicRenderingInfo& info,
                    std::optional<vulkan::DepthBiasInfo> depth_bias,
                    std::vector<uint32_t> vertex_shader = DepthOnlyVertexShader(),
                    std::vector<uint32_t> fragment_shader = {})
      : VulkanRender(context, info),
        depth_bias_(depth_bias),
        vertex_shader_(std::move(vertex_shader)),
        fragment_shader_(std::move(fragment_shader)) {}

  void Draw(VkCommandBuffer command_buffer, const DepthRamp& ramp, const VkRect2D& scissor) {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkViewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(kSize),
                              .height = static_cast<float>(kSize),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(DepthRamp), &ramp);
    vkCmdDraw(command_buffer, 3, 1, 0, 0);
  }

 private:
  VkCullModeFlags SetCullMode() const override { return VK_CULL_MODE_NONE; }
  VkBool32 SetDepthTesting() const override { return VK_TRUE; }
  VkBool32 SetDepthWriting() const override { return VK_TRUE; }
  std::optional<vulkan::DepthBiasInfo> SetDepthBias() const override { return depth_bias_; }

  std::vector<vulkan::BindingInfo> GetBindingInfo() const override { return {}; }
  std::vector<VkPushConstantRange> GetPushConstantRanges() const override {
    return {VkPushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(DepthRamp)}};
  }
  const std::vector<uint32_t> LoadVertexShader() const override { return vertex_shader_; }
  const std::vector<uint32_t> LoadFragmentShader() const override { return fragment_shader_; }
  std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const override {
    return {};
  }
  std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const override {
    return {};
  }

  std::optional<vulkan::DepthBiasInfo> depth_bias_;
  std::vector<uint32_t> vertex_shader_;
  std::vector<uint32_t> fragment_shader_;
};

struct DepthDraw {
  DepthOnlyPipeline* pipeline;
  DepthRamp ramp;
  VkRect2D scissor;
};

class VulkanDepthOnlyRenderTest : public testing::Test {
 protected:
  void SetUp() override {
    context_.Init();
    rendering_.emplace(&context_);
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context_.physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(context_.physical_device, &count, families.data());
    const uint32_t family = context_.GetQueueFamilyIndices().compute_family.value();
    if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      GTEST_SKIP() << "Compute queue family cannot record graphics work";
    }
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(context_.physical_device, kDepthFormat, &properties);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    if ((properties.optimalTilingFeatures & required) != required) {
      GTEST_SKIP() << "D32_SFLOAT depth attachments cannot be copied on this device";
    }
  }

  std::vector<float> Render(const std::vector<DepthDraw>& draws) {
    vulkan::VulkanImage depth(&context_, kSize, kSize, kDepthFormat,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              VK_IMAGE_ASPECT_DEPTH_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(kSize) * kSize * sizeof(float);
    vulkan::VulkanBuffer readback(
        &context_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vulkan::VulkanCommandBuffer command(&context_);
    command.BeginCommandBuffer();
    const VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                        .baseMipLevel = 0,
                                        .levelCount = 1,
                                        .baseArrayLayer = 0,
                                        .layerCount = 1};
    const VkImageMemoryBarrier to_attachment{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = depth.image,
        .subresourceRange = range};
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_attachment);

    const VkClearValue clear{.depthStencil = {.depth = 1.0f, .stencil = 0}};
    rendering_->BeginDepthRendering(command.buffer(), depth.image_view,
                                   VkExtent2D{.width = kSize, .height = kSize}, clear);
    for (const DepthDraw& draw : draws) {
      draw.pipeline->Draw(command.buffer(), draw.ramp, draw.scissor);
    }
    rendering_->EndDynamicRendering(command.buffer());

    const VkImageMemoryBarrier to_transfer{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = depth.image,
        .subresourceRange = range};
    vkCmdPipelineBarrier(command.buffer(),
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &to_transfer);
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = kSize, .height = kSize, .depth = 1}};
    vkCmdCopyImageToBuffer(command.buffer(), depth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &region);
    const VkMemoryBarrier to_host{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .pNext = nullptr,
                                  .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, nullptr, 0, nullptr);

    vulkan::VulkanFence fence(&context_);
    fence.Reset();
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));

    std::vector<float> texels(static_cast<size_t>(kSize) * kSize);
    readback.MapData([&](void* data) { std::memcpy(texels.data(), data, bytes); });
    return texels;
  }

  static VkRect2D Scissor(uint32_t x, uint32_t width) {
    return VkRect2D{.offset = {.x = static_cast<int32_t>(x), .y = 0},
                    .extent = {.width = width, .height = kSize}};
  }

  static vulkan::DynamicRenderingInfo DepthOnlyInfo() {
    vulkan::DynamicRenderingInfo info{};
    info.depth_format = kDepthFormat;
    return info;
  }

  vulkan::VulkanContext context_{true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE};
  std::optional<vulkan::VulkanDynamicRendering> rendering_;
};

TEST_F(VulkanDepthOnlyRenderTest, WritesAndTestsDepthWithoutColorOrFragmentStage) {
  DepthOnlyPipeline pipeline(&context_, DepthOnlyInfo(), std::nullopt);
  pipeline.Init();

  const std::vector<float> depth =
      Render({DepthDraw{.pipeline = &pipeline,
                        .ramp = {.left = 0.25f, .right = 0.25f},
                        .scissor = Scissor(0, kSize / 2)},
              DepthDraw{.pipeline = &pipeline,
                        .ramp = {.left = 0.5f, .right = 0.5f},
                        .scissor = Scissor(0, kSize)},
              DepthDraw{.pipeline = &pipeline,
                        .ramp = {.left = 0.75f, .right = 0.75f},
                        .scissor = Scissor(kSize / 4, kSize / 2)}});

  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const float expected = x < kSize / 2 ? 0.25f : 0.5f;
      ASSERT_EQ(depth[static_cast<size_t>(y) * kSize + x], expected) << x << "," << y;
    }
  }

  const std::vector<float> cleared =
      Render({DepthDraw{.pipeline = &pipeline,
                        .ramp = {.left = 0.25f, .right = 0.25f},
                        .scissor = Scissor(kSize - 4, 4)}});
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const float expected = x >= kSize - 4 ? 0.25f : 1.0f;
      ASSERT_EQ(cleared[static_cast<size_t>(y) * kSize + x], expected) << x << "," << y;
    }
  }
}

TEST_F(VulkanDepthOnlyRenderTest, AppliesConstantAndSlopeDepthBias) {
  DepthOnlyPipeline unbiased(&context_, DepthOnlyInfo(), std::nullopt);
  DepthOnlyPipeline constant(&context_, DepthOnlyInfo(),
                             vulkan::DepthBiasInfo{.constant_factor = 4096.0f,
                                                   .slope_factor = 0.0f});
  DepthOnlyPipeline slope(&context_, DepthOnlyInfo(),
                          vulkan::DepthBiasInfo{.constant_factor = 0.0f, .slope_factor = 2.0f});
  unbiased.Init();
  constant.Init();
  slope.Init();

  const DepthRamp flat{.left = 0.5f, .right = 0.5f};
  const DepthRamp ramp{.left = 0.2f, .right = 0.6f};
  const std::vector<float> flat_reference =
      Render({DepthDraw{.pipeline = &unbiased, .ramp = flat, .scissor = Scissor(0, kSize)}});
  const std::vector<float> flat_biased =
      Render({DepthDraw{.pipeline = &constant, .ramp = flat, .scissor = Scissor(0, kSize)}});
  const std::vector<float> ramp_reference =
      Render({DepthDraw{.pipeline = &unbiased, .ramp = ramp, .scissor = Scissor(0, kSize)}});
  const std::vector<float> ramp_biased =
      Render({DepthDraw{.pipeline = &slope, .ramp = ramp, .scissor = Scissor(0, kSize)}});

  const float slope_per_pixel = (ramp.right - ramp.left) / static_cast<float>(kSize);
  const float expected_slope_bias = 2.0f * slope_per_pixel;
  const float constant_upper_bound = 4096.0f * std::ldexp(1.0f, -22);
  const float constant_bias = flat_biased[0] - flat_reference[0];
  EXPECT_GT(constant_bias, 0.0f);
  EXPECT_LE(constant_bias, constant_upper_bound);
  for (size_t i = 0; i < flat_reference.size(); ++i) {
    ASSERT_EQ(flat_reference[i], 0.5f) << i;
    ASSERT_EQ(flat_biased[i], flat_biased[0]) << i;
  }
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const size_t index = static_cast<size_t>(y) * kSize + x;
      const float center = ramp.left + slope_per_pixel * (static_cast<float>(x) + 0.5f);
      ASSERT_NEAR(ramp_reference[index], center, 1e-5f) << x << "," << y;
      ASSERT_NEAR(ramp_biased[index] - ramp_reference[index], expected_slope_bias,
                  expected_slope_bias * 0.05f)
          << x << "," << y;
    }
  }
}

TEST_F(VulkanDepthOnlyRenderTest, RejectsInvalidDepthOnlyConfigurations) {
  DepthOnlyPipeline no_attachments(&context_, vulkan::DynamicRenderingInfo{}, std::nullopt);
  EXPECT_THROW(no_attachments.Init(), std::invalid_argument);

  vulkan::DynamicRenderingInfo color_info = DepthOnlyInfo();
  color_info.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
  DepthOnlyPipeline missing_fragment(&context_, color_info, std::nullopt);
  EXPECT_THROW(missing_fragment.Init(), std::invalid_argument);

  DepthOnlyPipeline missing_vertex(&context_, DepthOnlyInfo(), std::nullopt, {});
  EXPECT_THROW(missing_vertex.Init(), std::invalid_argument);

  DepthOnlyPipeline infinite_bias(
      &context_, DepthOnlyInfo(),
      vulkan::DepthBiasInfo{.constant_factor = std::numeric_limits<float>::infinity(),
                            .slope_factor = 0.0f});
  EXPECT_THROW(infinite_bias.Init(), std::invalid_argument);

  DepthOnlyPipeline nan_bias(
      &context_, DepthOnlyInfo(),
      vulkan::DepthBiasInfo{.constant_factor = 0.0f,
                            .slope_factor = std::numeric_limits<float>::quiet_NaN()});
  EXPECT_THROW(nan_bias.Init(), std::invalid_argument);

  vulkan::DynamicRenderingInfo two_colors = DepthOnlyInfo();
  two_colors.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
  DepthOnlyPipeline too_many_colors(&context_, two_colors, std::nullopt);
  EXPECT_THROW(too_many_colors.Init(), std::runtime_error);

  vulkan::VulkanImage depth(&context_, kSize, kSize, kDepthFormat,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanCommandBuffer command(&context_);
  const VkClearValue clear{.depthStencil = {.depth = 1.0f, .stencil = 0}};
  const VkExtent2D extent{.width = kSize, .height = kSize};
  EXPECT_THROW(rendering_->BeginDepthRendering(command.buffer(), VK_NULL_HANDLE, extent, clear),
               std::invalid_argument);
  EXPECT_THROW(rendering_->BeginDepthRendering(VK_NULL_HANDLE, depth.image_view, extent, clear),
               std::invalid_argument);
  EXPECT_THROW(rendering_->BeginDepthRendering(command.buffer(), depth.image_view,
                                               VkExtent2D{.width = 0, .height = kSize}, clear),
               std::invalid_argument);
  EXPECT_THROW(rendering_->BeginDepthRendering(command.buffer(), depth.image_view,
                                               VkExtent2D{.width = kSize, .height = 0}, clear),
               std::invalid_argument);
}

}
}
}
