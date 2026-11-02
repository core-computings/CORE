#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

#include "VulkanImage.h"
#include "VulkanRender.h"
#include "VulkanRenderPass.h"
#include "VulkanSampler.h"
#include "VulkanUtils.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/hash.hpp"

namespace core {

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 tex_coord;

  bool operator==(const Vertex& other) const {
    return pos == other.pos && color == other.color && tex_coord == other.tex_coord;
  }
};

class RenderModel : public core::vulkan::VulkanRender {
 public:
  RenderModel(core::vulkan::VulkanContext* context, core::vulkan::VulkanRenderPass* render_pass);

  void Init() override;
  void Init(const std::string& image_path, const std::string& model_path);
  void Render(VkCommandBuffer command_buffer, VkExtent2D extent);

  void UpdateUniformBuffer(const int width, const int height, const glm::mat4& view_matrix);

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
  void CreateTextureImage(const std::string& image_path);

  struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 project;
  } uniform_data_;

  void LoadModel(const std::string& model_path);
  void CreateBuffers();

  // vertex
  std::vector<Vertex> vertices_;
  // index
  std::vector<uint32_t> indices_;

  core::vulkan::VulkanBuffer vertex_buffer_local_;

  core::vulkan::VulkanBuffer index_buffer_local_;

  // uniform buffer
  core::vulkan::VulkanBuffer uniform_buffer_;

  // texture image
  core::vulkan::VulkanImage texture_image_;
  core::vulkan::VulkanSampler sampler_;
};

}  // namespace core

namespace std {
template <>
struct hash<core::Vertex> {
  size_t operator()(core::Vertex const& vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
           (hash<glm::vec2>()(vertex.tex_coord) << 1);
  }
};
}  // namespace std
