#pragma once

#include <vector>

#include "VulkanBuffer.h"
#include "VulkanRender.h"
#include "VulkanUtils.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"

namespace core {

// Renders a procedurally-built car composed of axis-aligned boxes: body, cabin
// and four wheels. The geometry is baked once at construction time so each
// box's per-part transform is folded into the vertex positions.
class RenderCar : public core::vulkan::VulkanRender {
 public:
  RenderCar(core::vulkan::VulkanContext* context,
            const core::vulkan::DynamicRenderingInfo& dynamic_rendering_info);

  void Init() override;
  void Render(VkCommandBuffer command_buffer, VkExtent2D extent);

  void UpdateUniformBuffer(const glm::mat4& model, const glm::mat4& view, const glm::mat4& project);

  const core::vulkan::VulkanBuffer& vertex_buffer() const { return vertex_buffer_local_; }
  const core::vulkan::VulkanBuffer& index_buffer() const { return index_buffer_local_; }
  uint32_t index_count() const { return static_cast<uint32_t>(indices_.size()); }
  static constexpr uint32_t kVertexStride = 9 * sizeof(float);

 protected:
  VkCullModeFlags SetCullMode() const override { return VK_CULL_MODE_BACK_BIT; }
  VkFrontFace SetFrontFace() const override { return VK_FRONT_FACE_COUNTER_CLOCKWISE; }
  VkBool32 SetDepthTesting() const override { return VK_TRUE; }
  VkBool32 SetDepthWriting() const override { return VK_TRUE; }

  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t> LoadVertexShader() const override;
  const std::vector<uint32_t> LoadFragmentShader() const override;
  std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const override;
  std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const override;

 private:
  struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
  };

  struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 project;
  };

  void BuildGeometry();
  void CreateBuffers();
  // Emits 24 vertices + 36 indices for one axis-aligned box (center, size, color).
  void AppendBox(const glm::vec3& center, const glm::vec3& size, const glm::vec3& color);

  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;

  core::vulkan::VulkanBuffer vertex_buffer_staging_;
  core::vulkan::VulkanBuffer vertex_buffer_local_;
  core::vulkan::VulkanBuffer index_buffer_staging_;
  core::vulkan::VulkanBuffer index_buffer_local_;
  core::vulkan::VulkanBuffer uniform_buffer_;
};

}  // namespace core
