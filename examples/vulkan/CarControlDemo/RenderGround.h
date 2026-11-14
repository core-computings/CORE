#pragma once

#include <string>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanImage.h"
#include "VulkanRender.h"
#include "VulkanSampler.h"
#include "VulkanUtils.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"

namespace core {

// Big ground plane at y=0, textured with a tiled diffuse map and faded out
// toward the horizon so the plane's edge doesn't read as a hard seam against
// the skybox.
class RenderGround : public core::vulkan::VulkanRender {
 public:
  RenderGround(core::vulkan::VulkanContext* context,
               const core::vulkan::DynamicRenderingInfo& dynamic_rendering_info);

  void Init() override;
  void Init(const std::string& texture_path, VkImageView shadow_map_view, VkSampler shadow_sampler);
  void Render(VkCommandBuffer command_buffer, VkExtent2D extent);

  void UpdateUniformBuffer(const glm::mat4& view, const glm::mat4& project,
                           const glm::mat4& light_view_projection);

 protected:
  VkCullModeFlags SetCullMode() const override { return VK_CULL_MODE_NONE; }
  VkFrontFace SetFrontFace() const override { return VK_FRONT_FACE_COUNTER_CLOCKWISE; }
  VkBool32 SetDepthTesting() const override { return VK_TRUE; }
  VkBool32 SetDepthWriting() const override { return VK_TRUE; }

  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t> LoadVertexShader() const override;
  const std::vector<uint32_t> LoadFragmentShader() const override;
  std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const override;
  std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const override;

 private:
  struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 project;
    glm::mat4 light_view_projection;
  };

  void CreateTextureImage(const std::string& texture_path);
  void CreateBuffers();

  static constexpr float kHalf = 80.0f;
  const std::vector<glm::vec3> vertices_ = {
      {-kHalf, 0.0f, -kHalf},
      {kHalf, 0.0f, -kHalf},
      {kHalf, 0.0f, kHalf},
      {-kHalf, 0.0f, kHalf},
  };
  const std::vector<uint16_t> indices_ = {0, 1, 2, 0, 2, 3};

  core::vulkan::VulkanBuffer vertex_buffer_staging_;
  core::vulkan::VulkanBuffer vertex_buffer_local_;
  core::vulkan::VulkanBuffer index_buffer_staging_;
  core::vulkan::VulkanBuffer index_buffer_local_;
  core::vulkan::VulkanBuffer uniform_buffer_;

  core::vulkan::VulkanImage texture_image_;
  core::vulkan::VulkanSampler sampler_;
  VkImageView shadow_map_view_ = VK_NULL_HANDLE;
  VkSampler shadow_sampler_ = VK_NULL_HANDLE;
};

}  // namespace core
