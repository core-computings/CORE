#include "RenderCar.h"

#include <cstring>

namespace core {

RenderCar::RenderCar(core::vulkan::VulkanContext* context,
                     const core::vulkan::DynamicRenderingInfo& dynamic_rendering_info)
    : core::vulkan::VulkanRender(context, dynamic_rendering_info) {
  BuildGeometry();
  CreateBuffers();
}

void RenderCar::Init() {
  core::vulkan::VulkanRender::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);

  vertex_buffer_staging_.MapData([this](void* data) {
    memcpy(data, vertices_.data(), sizeof(vertices_[0]) * vertices_.size());
  });
  index_buffer_staging_.MapData(
      [this](void* data) { memcpy(data, indices_.data(), sizeof(indices_[0]) * indices_.size()); });
  vertex_buffer_staging_.CopyToBuffer(vertex_buffer_local_);
  index_buffer_staging_.CopyToBuffer(index_buffer_local_);

  uniform_buffer_.MapData([](void* data) {
    UniformBufferObject init{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)};
    memcpy(data, &init, sizeof(UniformBufferObject));
  });
}

void RenderCar::Render(VkCommandBuffer command_buffer, VkExtent2D extent) {
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
  vkCmdBindIndexBuffer(command_buffer, index_buffer_local_.buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
}

void RenderCar::UpdateUniformBuffer(const glm::mat4& model, const glm::mat4& view,
                                    const glm::mat4& project) {
  uniform_buffer_.MapData([&](void* data) {
    UniformBufferObject ubo{model, view, project};
    ubo.project[1][1] *= -1;  // Vulkan Y-flip
    memcpy(data, &ubo, sizeof(UniformBufferObject));
  });
}

std::vector<core::vulkan::BindingInfo> RenderCar::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}};
}

const std::vector<uint32_t> RenderCar::LoadVertexShader() const {
  static const std::vector<uint32_t> shader_code =
#include "Car.vert.spv"
      ;
  return shader_code;
}

const std::vector<uint32_t> RenderCar::LoadFragmentShader() const {
  static const std::vector<uint32_t> shader_code =
#include "Car.frag.spv"
      ;
  return shader_code;
}

std::vector<VkVertexInputBindingDescription> RenderCar::GetVertexBindingDescriptions() const {
  VkVertexInputBindingDescription b{};
  b.binding = 0;
  static_assert(sizeof(Vertex) == kVertexStride);
  b.stride = kVertexStride;
  b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return {b};
}

std::vector<VkVertexInputAttributeDescription> RenderCar::GetVertexAttributeDescriptions() const {
  std::vector<VkVertexInputAttributeDescription> a(3);
  a[0].binding = 0;
  a[0].location = 0;
  a[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  a[0].offset = offsetof(Vertex, pos);

  a[1].binding = 0;
  a[1].location = 1;
  a[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  a[1].offset = offsetof(Vertex, normal);

  a[2].binding = 0;
  a[2].location = 2;
  a[2].format = VK_FORMAT_R32G32B32_SFLOAT;
  a[2].offset = offsetof(Vertex, color);
  return a;
}

void RenderCar::AppendBox(const glm::vec3& center, const glm::vec3& size, const glm::vec3& color) {
  const glm::vec3 h = size * 0.5f;
  const glm::vec3 normals[6] = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
  };
  // For each face, four corners in CCW order viewed along +normal, in
  // box-local coordinates with extents ±h.
  const glm::vec3 face_corners[6][4] = {
      {{h.x, -h.y, -h.z}, {h.x, h.y, -h.z}, {h.x, h.y, h.z}, {h.x, -h.y, h.z}},      // +X
      {{-h.x, -h.y, h.z}, {-h.x, h.y, h.z}, {-h.x, h.y, -h.z}, {-h.x, -h.y, -h.z}},  // -X
      {{-h.x, h.y, -h.z}, {-h.x, h.y, h.z}, {h.x, h.y, h.z}, {h.x, h.y, -h.z}},      // +Y
      {{-h.x, -h.y, h.z}, {-h.x, -h.y, -h.z}, {h.x, -h.y, -h.z}, {h.x, -h.y, h.z}},  // -Y
      {{-h.x, -h.y, h.z}, {h.x, -h.y, h.z}, {h.x, h.y, h.z}, {-h.x, h.y, h.z}},      // +Z
      {{h.x, -h.y, -h.z}, {-h.x, -h.y, -h.z}, {-h.x, h.y, -h.z}, {h.x, h.y, -h.z}},  // -Z
  };

  for (int f = 0; f < 6; ++f) {
    const uint32_t base = static_cast<uint32_t>(vertices_.size());
    for (int c = 0; c < 4; ++c) {
      vertices_.push_back({center + face_corners[f][c], normals[f], color});
    }
    indices_.push_back(base + 0);
    indices_.push_back(base + 1);
    indices_.push_back(base + 2);
    indices_.push_back(base + 0);
    indices_.push_back(base + 2);
    indices_.push_back(base + 3);
  }
}

void RenderCar::BuildGeometry() {
  // Convention: +Z is "forward" (where the headlights point), +Y is up.
  // Wheels (height 0.6, center y=0.3) sit with their bottoms on y=0.
  const glm::vec3 body_color(0.85f, 0.20f, 0.20f);
  const glm::vec3 cabin_color(0.18f, 0.22f, 0.30f);
  const glm::vec3 wheel_color(0.08f, 0.08f, 0.10f);

  // Body: long red box.
  AppendBox({0.0f, 0.50f, 0.0f}, {1.8f, 0.55f, 4.0f}, body_color);
  // Cabin: shorter box, shifted toward the rear to leave a hood at the front.
  AppendBox({0.0f, 1.05f, -0.4f}, {1.4f, 0.55f, 2.2f}, cabin_color);
  // Four wheels: bulge slightly outside the body so they read as tires.
  const float wheel_dx = 1.0f;
  const float wheel_dz = 1.35f;
  AppendBox({wheel_dx, 0.30f, wheel_dz}, {0.30f, 0.60f, 0.70f}, wheel_color);
  AppendBox({-wheel_dx, 0.30f, wheel_dz}, {0.30f, 0.60f, 0.70f}, wheel_color);
  AppendBox({wheel_dx, 0.30f, -wheel_dz}, {0.30f, 0.60f, 0.70f}, wheel_color);
  AppendBox({-wheel_dx, 0.30f, -wheel_dz}, {0.30f, 0.60f, 0.70f}, wheel_color);
}

void RenderCar::CreateBuffers() {
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
