#include "RenderCubeMap.h"

#include <bit>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace core {

RenderCubeMap::RenderCubeMap(core::vulkan::VulkanContext* context,
                             const core::vulkan::DynamicRenderingInfo& dynamic_rendering_info)
    : core::vulkan::VulkanRender(context, dynamic_rendering_info), sampler_(context) {}

void RenderCubeMap::Init() {
  core::vulkan::VulkanRender::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateCombinedImageSamplerDescriptorSet(1, cube_map_image_.image_view, sampler_.sampler);
  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);

  // Prepare vertex buffer
  vertex_buffer_staging_.MapData([this](void* data) {
    memcpy(data, skybox_vertices_.data(), sizeof(float) * skybox_vertices_.size());
  });
  vertex_buffer_staging_.CopyToBuffer(vertex_buffer_local_);

  // uniform buffer
  uniform_buffer_.MapData([this](void* data) {
    uniform_data_.view = glm::mat4(1.0f);
    uniform_data_.project = glm::mat4(1.0f);
    memcpy(data, &uniform_data_, sizeof(UniformBufferObject));
  });
}

void RenderCubeMap::Render(VkCommandBuffer command_buffer, VkExtent2D extent) {
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
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  // 36 vertices
  vkCmdDraw(command_buffer, 36, 1, 0, 0);
}

std::vector<core::vulkan::BindingInfo> RenderCubeMap::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
          {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}};
}

const std::vector<uint32_t> RenderCubeMap::LoadVertexShader() const {
  static const std::vector<uint32_t> shader_code =
#include "CubeMap.vert.spv"
      ;
  return shader_code;
}

const std::vector<uint32_t> RenderCubeMap::LoadFragmentShader() const {
  static const std::vector<uint32_t> shader_code =
#include "CubeMap.frag.spv"
      ;
  return shader_code;
}

std::vector<VkVertexInputBindingDescription> RenderCubeMap::GetVertexBindingDescriptions() const {
  VkVertexInputBindingDescription binding_description{};
  binding_description.binding = 0;
  binding_description.stride = sizeof(float) * 3;  // only position
  binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  return {binding_description};
}

std::vector<VkVertexInputAttributeDescription> RenderCubeMap::GetVertexAttributeDescriptions()
    const {
  VkVertexInputAttributeDescription attribute_description{};
  attribute_description.binding = 0;
  attribute_description.location = 0;
  attribute_description.format = VK_FORMAT_R32G32B32_SFLOAT;
  attribute_description.offset = 0;

  return {attribute_description};
}

void RenderCubeMap::Init(const std::string& image_path) {
  CreateTextureImage(image_path);
  CreateBuffers();
  Init();
}

void RenderCubeMap::CreateBuffers() {
  const VkDeviceSize vertex_buffer_size = skybox_vertices_.size() * sizeof(float);

  vertex_buffer_staging_ = core::vulkan::VulkanBuffer(
      context_, vertex_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vertex_buffer_local_ = core::vulkan::VulkanBuffer(
      context_, vertex_buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  uniform_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void RenderCubeMap::UpdateUniformBuffer(const int width, const int height,
                                        const glm::mat4& view_matrix) {
  // TODO: maintain a persistent mapping pointer to avoid mapping every time
  uniform_buffer_.MapData([this, width, height, &view_matrix](void* data) {
    // Keep the skybox centered on the camera by discarding translation.
    uniform_data_.view = glm::mat4(glm::mat3(view_matrix));
    uniform_data_.project =
        glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
    uniform_data_.project[1][1] *= -1;  // Invert Y for Vulkan

    memcpy(data, &uniform_data_, sizeof(UniformBufferObject));
  });
}

void RenderCubeMap::CreateTextureImage(const std::string& image_path) {
  int texture_width = 0;
  int texture_height = 0;
  int texture_channels = 0;
  const int kTextureChannels = STBI_rgb_alpha;
  stbi_uc* pixels = stbi_load(image_path.c_str(), &texture_width, &texture_height,
                              &texture_channels, kTextureChannels);
  if (!pixels) {
    throw std::runtime_error("failed to load cube map image!");
  }

  // Load and process cubemap
  core::io::Bitmap in(texture_width, texture_height, kTextureChannels,
                      BitmapFormat::BitmapFormat_Uint8, pixels);
  core::io::Bitmap out = core::io::ConvertVerticalCrossToCubeMapFaces(in);
  printf("Processed cube map image: width=%d, height=%d, channels=%d\n", out.width, out.height,
         out.depth);

  const uint32_t face_size = static_cast<uint32_t>(out.width);
  const uint32_t mip_levels = std::bit_width(face_size);
  VkDeviceSize image_size = out.width * out.height * out.depth;

  core::vulkan::VulkanBuffer staging_buffer(
      context_, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  staging_buffer.MapData([&out, image_size](void* data) {
    memcpy(data, out.pixel.data(), static_cast<size_t>(image_size));
  });

  stbi_image_free(pixels);

  cube_map_image_ =
      core::vulkan::VulkanImage(context_, out.width, out.height, VK_FORMAT_R8G8B8A8_SRGB,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                VK_IMAGE_TILING_OPTIMAL, mip_levels, VK_SAMPLE_COUNT_1_BIT,
                                VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 6);

  const vulkan::VulkanCommandBuffer upload =
      vulkan::VulkanCommandBuffer::BeginOneTimeCommands(context_);
  cube_map_image_.TransitionImageLayout(upload.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_FORMAT_R8G8B8A8_SRGB, mip_levels, 6);
  staging_buffer.CopyToImage(upload.buffer(), cube_map_image_, static_cast<uint32_t>(out.width),
                             static_cast<uint32_t>(out.height), 6);
  cube_map_image_.GenerateMipmaps(upload.buffer());
  upload.EndOneTimeCommands();
}

}  // namespace core
