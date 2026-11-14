#include "RenderGround.h"

#include <cstring>
#include <stdexcept>

// stb_image's implementation lives in RenderSkybox.cpp; include the header
// here without the implementation define.
#include <stb_image.h>

namespace core {

RenderGround::RenderGround(core::vulkan::VulkanContext* context,
                           const core::vulkan::DynamicRenderingInfo& dynamic_rendering_info)
    : core::vulkan::VulkanRender(context, dynamic_rendering_info), sampler_(context) {}

void RenderGround::Init() {
  core::vulkan::VulkanRender::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateCombinedImageSamplerDescriptorSet(1, texture_image_.image_view, sampler_.sampler);
  CreateCombinedImageSamplerDescriptorSet(2, shadow_map_view_, shadow_sampler_);
  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);

  vertex_buffer_staging_.MapData([this](void* data) {
    memcpy(data, vertices_.data(), sizeof(vertices_[0]) * vertices_.size());
  });
  index_buffer_staging_.MapData(
      [this](void* data) { memcpy(data, indices_.data(), sizeof(indices_[0]) * indices_.size()); });
  vertex_buffer_staging_.CopyToBuffer(vertex_buffer_local_);
  index_buffer_staging_.CopyToBuffer(index_buffer_local_);

  uniform_buffer_.MapData([](void* data) {
    UniformBufferObject init{.view = glm::mat4(1.0f),
                             .project = glm::mat4(1.0f),
                             .light_view_projection = glm::mat4(1.0f)};
    memcpy(data, &init, sizeof(UniformBufferObject));
  });
}

void RenderGround::Init(const std::string& texture_path, VkImageView shadow_map_view,
                        VkSampler shadow_sampler) {
  if (shadow_map_view == VK_NULL_HANDLE || shadow_sampler == VK_NULL_HANDLE) {
    throw std::invalid_argument("Ground rendering requires a shadow map and sampler");
  }
  shadow_map_view_ = shadow_map_view;
  shadow_sampler_ = shadow_sampler;
  CreateTextureImage(texture_path);
  CreateBuffers();
  Init();
}

void RenderGround::Render(VkCommandBuffer command_buffer, VkExtent2D extent) {
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(extent.width);
  viewport.height = static_cast<float>(extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = extent;
  vkCmdSetScissor(command_buffer, 0, 1, &scissor);

  const VkBuffer vertex_buffers[] = {vertex_buffer_local_.buffer};
  const VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
  vkCmdBindIndexBuffer(command_buffer, index_buffer_local_.buffer, 0, VK_INDEX_TYPE_UINT16);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
}

void RenderGround::UpdateUniformBuffer(const glm::mat4& view, const glm::mat4& project,
                                       const glm::mat4& light_view_projection) {
  uniform_buffer_.MapData([&](void* data) {
    UniformBufferObject ubo{
        .view = view, .project = project, .light_view_projection = light_view_projection};
    ubo.project[1][1] *= -1;  // Vulkan Y-flip
    memcpy(data, &ubo, sizeof(UniformBufferObject));
  });
}

std::vector<core::vulkan::BindingInfo> RenderGround::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
          {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
          {.binding = 2,
           .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           .shader_stage = VK_SHADER_STAGE_FRAGMENT_BIT}};
}

const std::vector<uint32_t> RenderGround::LoadVertexShader() const {
  static const std::vector<uint32_t> shader_code =
#include "Ground.vert.spv"
      ;
  return shader_code;
}

const std::vector<uint32_t> RenderGround::LoadFragmentShader() const {
  static const std::vector<uint32_t> shader_code =
#include "Ground.frag.spv"
      ;
  return shader_code;
}

std::vector<VkVertexInputBindingDescription> RenderGround::GetVertexBindingDescriptions() const {
  VkVertexInputBindingDescription b{};
  b.binding = 0;
  b.stride = sizeof(glm::vec3);
  b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return {b};
}

std::vector<VkVertexInputAttributeDescription> RenderGround::GetVertexAttributeDescriptions()
    const {
  VkVertexInputAttributeDescription a{};
  a.binding = 0;
  a.location = 0;
  a.format = VK_FORMAT_R32G32B32_SFLOAT;
  a.offset = 0;
  return {a};
}

void RenderGround::CreateTextureImage(const std::string& texture_path) {
  int w = 0;
  int h = 0;
  int c = 0;
  stbi_uc* pixels = stbi_load(texture_path.c_str(), &w, &h, &c, STBI_rgb_alpha);
  if (!pixels) {
    throw std::runtime_error("failed to load ground texture: " + texture_path);
  }
  const VkDeviceSize image_size = static_cast<VkDeviceSize>(w) * h * 4;

  core::vulkan::VulkanBuffer staging_buffer(
      context_, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  staging_buffer.MapData(
      [&](void* data) { memcpy(data, pixels, static_cast<size_t>(image_size)); });
  stbi_image_free(pixels);

  texture_image_ = core::vulkan::VulkanImage(
      context_, static_cast<uint32_t>(w), static_cast<uint32_t>(h), VK_FORMAT_R8G8B8A8_SRGB,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL);
  texture_image_.TransitionImageLayout(
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_FORMAT_R8G8B8A8_SRGB);
  staging_buffer.CopyToImage(texture_image_, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
  texture_image_.TransitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_FORMAT_R8G8B8A8_SRGB);
}

void RenderGround::CreateBuffers() {
  const VkDeviceSize v_size = sizeof(vertices_[0]) * vertices_.size();
  const VkDeviceSize i_size = sizeof(indices_[0]) * indices_.size();

  vertex_buffer_staging_ = core::vulkan::VulkanBuffer(
      context_, v_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vertex_buffer_local_ = core::vulkan::VulkanBuffer(
      context_, v_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  index_buffer_staging_ = core::vulkan::VulkanBuffer(
      context_, i_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  index_buffer_local_ = core::vulkan::VulkanBuffer(
      context_, i_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  uniform_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

}  // namespace core
