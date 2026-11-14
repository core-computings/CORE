#include "RenderCarShadow.h"

#include <stdexcept>

namespace core {

RenderCarShadow::RenderCarShadow(core::vulkan::VulkanContext* context, VkFormat depth_format,
                                 uint32_t size)
    : core::vulkan::VulkanRender(context, DepthOnlyInfo(depth_format)),
      depth_format_(depth_format),
      size_(size),
      dynamic_rendering_(context) {
  if (size_ == 0) {
    throw std::invalid_argument("Shadow map size must be positive");
  }
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(context_->physical_device, depth_format_, &properties);
  const VkFormatFeatureFlags required =
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  if ((properties.optimalTilingFeatures & required) != required) {
    throw std::runtime_error("Shadow map format cannot be rendered and sampled");
  }
  const bool linear_filter =
      (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

  shadow_map_ = core::vulkan::VulkanImage(
      context_, size_, size_, depth_format_,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_ASPECT_DEPTH_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  const VkFilter filter = linear_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  const VkSamplerCreateInfo sampler_info{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .magFilter = filter,
      .minFilter = filter,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_FALSE,
      .maxAnisotropy = 1.0f,
      .compareEnable = VK_TRUE,
      .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
      .minLod = 0.0f,
      .maxLod = 0.0f,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      .unnormalizedCoordinates = VK_FALSE,
  };
  VK_CHECK(vkCreateSampler(context_->logical_device, &sampler_info, nullptr, &sampler_));
}

RenderCarShadow::~RenderCarShadow() {
  if (sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(context_->logical_device, sampler_, nullptr);
  }
}

void RenderCarShadow::Init() {
  core::vulkan::VulkanRender::Init();

  const core::vulkan::VulkanCommandBuffer command =
      core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(context_);
  const VkImageMemoryBarrier to_shader_read{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = shadow_map_.image,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };
  vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                       &to_shader_read);
  command.EndOneTimeCommands();
}

void RenderCarShadow::Record(VkCommandBuffer command_buffer, const RenderCar& car,
                             const glm::mat4& light_model_view_projection) const {
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
      .image = shadow_map_.image,
      .subresourceRange = range,
  };
  vkCmdPipelineBarrier(
      command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0,
      nullptr, 0, nullptr, 1, &to_attachment);

  const VkExtent2D extent{.width = size_, .height = size_};
  const VkClearValue clear{.depthStencil = {.depth = 1.0f, .stencil = 0}};
  dynamic_rendering_.BeginDepthRendering(command_buffer, shadow_map_.image_view, extent, clear);
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  const VkViewport viewport{.x = 0.0f,
                            .y = 0.0f,
                            .width = static_cast<float>(size_),
                            .height = static_cast<float>(size_),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f};
  vkCmdSetViewport(command_buffer, 0, 1, &viewport);
  const VkRect2D scissor{.offset = {.x = 0, .y = 0}, .extent = extent};
  vkCmdSetScissor(command_buffer, 0, 1, &scissor);
  vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(glm::mat4), &light_model_view_projection);
  const VkBuffer vertex_buffer = car.vertex_buffer().buffer;
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
  vkCmdBindIndexBuffer(command_buffer, car.index_buffer().buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(command_buffer, car.index_count(), 1, 0, 0, 0);
  dynamic_rendering_.EndDynamicRendering(command_buffer);

  const VkImageMemoryBarrier to_shader_read{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = shadow_map_.image,
      .subresourceRange = range,
  };
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader_read);
}

std::vector<VkPushConstantRange> RenderCarShadow::GetPushConstantRanges() const {
  return {VkPushConstantRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(glm::mat4)}};
}

const std::vector<uint32_t> RenderCarShadow::LoadVertexShader() const {
  static const std::vector<uint32_t> shader_code =
#include "CarShadow.vert.spv"
      ;
  return shader_code;
}

std::vector<VkVertexInputBindingDescription> RenderCarShadow::GetVertexBindingDescriptions()
    const {
  return {VkVertexInputBindingDescription{.binding = 0,
                                          .stride = RenderCar::kVertexStride,
                                          .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
}

std::vector<VkVertexInputAttributeDescription> RenderCarShadow::GetVertexAttributeDescriptions()
    const {
  return {VkVertexInputAttributeDescription{
      .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0}};
}

core::vulkan::DynamicRenderingInfo RenderCarShadow::DepthOnlyInfo(VkFormat depth_format) {
  core::vulkan::DynamicRenderingInfo info{};
  info.depth_format = depth_format;
  return info;
}

}
