#include "RenderModel.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace core {

RenderModel::RenderModel(core::vulkan::VulkanContext* context,
                         core::vulkan::VulkanRenderPass* render_pass)
    : core::vulkan::VulkanRender(context, render_pass), sampler_(context) {}

void RenderModel::Init() {
  core::vulkan::VulkanRender::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateCombinedImageSamplerDescriptorSet(1, texture_image_.image_view, sampler_.sampler);
  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);

  vulkan::VulkanBuffer vertex_staging(
      context_, vertex_buffer_local_.Size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vulkan::VulkanBuffer index_staging(
      context_, index_buffer_local_.Size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vertex_staging.MapData([this](void* data) {
    memcpy(data, vertices_.data(), sizeof(vertices_[0]) * vertices_.size());
  });
  index_staging.MapData(
      [this](void* data) { memcpy(data, indices_.data(), sizeof(indices_[0]) * indices_.size()); });

  uniform_buffer_.MapData([this](void* data) {
    uniform_data_.model = glm::mat4(1.0f);
    uniform_data_.view = glm::mat4(1.0f);
    uniform_data_.project = glm::mat4(1.0f);
    memcpy(data, &uniform_data_, sizeof(UniformBufferObject));
  });

  const vulkan::BatchBufferCopy uploads[] = {
      {.source = &vertex_staging,
       .destination = &vertex_buffer_local_,
       .source_offset = 0,
       .destination_offset = 0,
       .size = VK_WHOLE_SIZE},
      {.source = &index_staging,
       .destination = &index_buffer_local_,
       .source_offset = 0,
       .destination_offset = 0,
       .size = VK_WHOLE_SIZE}};
  vulkan::VulkanBuffer::CopyBuffers(uploads);
}

void RenderModel::Init(const std::string& image_path, const std::string& model_path) {
  CreateTextureImage(image_path);
  LoadModel(model_path);
  CreateBuffers();
  Init();
}

void RenderModel::Render(VkCommandBuffer command_buffer, VkExtent2D extent) {
  const VkBuffer vertex_buffers[] = {vertex_buffer_local_.buffer};
  const VkDeviceSize offsets[] = {0};
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)extent.width;
  viewport.height = (float)extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = extent;
  vkCmdSetScissor(command_buffer, 0, 1, &scissor);

  vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
  vkCmdBindIndexBuffer(command_buffer, index_buffer_local_.buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
}

std::vector<core::vulkan::BindingInfo> RenderModel::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
          {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}};
}

const std::vector<uint32_t> RenderModel::LoadVertexShader() const {
  static const std::vector<uint32_t> shader_code =
#include "Model.vert.spv"
      ;
  return shader_code;
}

const std::vector<uint32_t> RenderModel::LoadFragmentShader() const {
  static const std::vector<uint32_t> shader_code =
#include "Model.frag.spv"
      ;
  return shader_code;
}

std::vector<VkVertexInputBindingDescription> RenderModel::GetVertexBindingDescriptions() const {
  VkVertexInputBindingDescription binding_description{};
  binding_description.binding = 0;
  binding_description.stride = sizeof(Vertex);
  binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  return {binding_description};
}

std::vector<VkVertexInputAttributeDescription> RenderModel::GetVertexAttributeDescriptions() const {
  std::vector<VkVertexInputAttributeDescription> attribute_descriptions(3);
  attribute_descriptions[0].binding = 0;
  attribute_descriptions[0].location = 0;
  attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attribute_descriptions[0].offset = offsetof(Vertex, pos);

  attribute_descriptions[1].binding = 0;
  attribute_descriptions[1].location = 1;
  attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attribute_descriptions[1].offset = offsetof(Vertex, color);

  attribute_descriptions[2].binding = 0;
  attribute_descriptions[2].location = 2;
  attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
  attribute_descriptions[2].offset = offsetof(Vertex, tex_coord);
  return attribute_descriptions;
}

void RenderModel::LoadModel(const std::string& model_path) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, model_path.c_str())) {
    throw std::runtime_error(err);
  }

  std::unordered_map<Vertex, uint32_t> unique_vertices{};

  for (const auto& shape : shapes) {
    for (const auto& index : shape.mesh.indices) {
      Vertex vertex{};

      vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]};

      vertex.tex_coord = {attrib.texcoords[2 * index.texcoord_index + 0],
                          1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

      vertex.color = {1.0f, 1.0f, 1.0f};

      if (unique_vertices.count(vertex) == 0) {
        unique_vertices[vertex] = static_cast<uint32_t>(vertices_.size());
        vertices_.push_back(vertex);
      }

      indices_.push_back(unique_vertices[vertex]);
    }
  }
  printf("Loaded model vertices: %zu, indices: %zu\n", vertices_.size(), indices_.size());
}

void RenderModel::CreateBuffers() {
  const VkDeviceSize vertex_buffer_size = sizeof(vertices_[0]) * vertices_.size();
  const VkDeviceSize index_buffer_size = sizeof(indices_[0]) * indices_.size();

  vertex_buffer_local_ = core::vulkan::VulkanBuffer(
      context_, vertex_buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  index_buffer_local_ = core::vulkan::VulkanBuffer(
      context_, index_buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  uniform_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void RenderModel::UpdateUniformBuffer(const int width, const int height,
                                      const glm::mat4& view_matrix) {
  // TODO: maintain a persistent mapping pointer to avoid mapping every time
  uniform_buffer_.MapData([this, width, height, &view_matrix](void* data) {
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    uniform_data_.model = model;
    uniform_data_.view = view_matrix;
    uniform_data_.project =
        glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
    uniform_data_.project[1][1] *= -1;  // Invert Y for Vulkan

    memcpy(data, &uniform_data_, sizeof(UniformBufferObject));
  });
}

void RenderModel::CreateTextureImage(const std::string& image_path) {
  int texture_width, texture_height, texture_channels;
  stbi_uc* pixels = stbi_load(image_path.c_str(), &texture_width, &texture_height,
                              &texture_channels, STBI_rgb_alpha);
  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }
  VkDeviceSize image_size = texture_width * texture_height * 4;

  core::vulkan::VulkanBuffer staging_buffer(
      context_, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  staging_buffer.MapData(
      [&pixels, image_size](void* data) { memcpy(data, pixels, static_cast<size_t>(image_size)); });

  stbi_image_free(pixels);

  texture_image_ =
      core::vulkan::VulkanImage(context_, texture_width, texture_height, VK_FORMAT_R8G8B8A8_SRGB,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Transition image layout and copy buffer to image
  // TODO: Use a single command buffer for all operations for higher throughput,
  texture_image_.TransitionImageLayout(
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_FORMAT_R8G8B8A8_SRGB);
  staging_buffer.CopyToImage(texture_image_, static_cast<uint32_t>(texture_width),
                             static_cast<uint32_t>(texture_height));
  texture_image_.TransitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_FORMAT_R8G8B8A8_SRGB);
}

}  // namespace core