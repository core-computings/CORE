#pragma once

#include <vulkan/vulkan.h>

#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"

namespace core {
namespace vulkan {

class VulkanBuffer;

class VulkanImage {
 public:
  VulkanImage() = default;
  VulkanImage(VulkanContext* context, const uint32_t width, const uint32_t height,
              const VkFormat format, const VkImageUsageFlags usage, const VkImageAspectFlags aspect,
              const VkMemoryPropertyFlags properties,
              const VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL, const uint32_t mip_levels = 1,
              const VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
              const VkImageCreateFlags flags = 0, const uint32_t layers = 1);
  ~VulkanImage();

  VulkanImage& operator=(VulkanImage&&);

  void TransitionImageLayout(const VkImageLayout old_layout, const VkImageLayout new_layout,
                             [[maybe_unused]] const VkFormat format, const uint32_t mip_levels = 1,
                             const uint32_t layers = 1);

  void TransitionImageLayout(VkCommandBuffer command_buffer, VkImageLayout old_layout,
                             VkImageLayout new_layout, VkFormat format, uint32_t mip_levels = 1,
                             uint32_t layers = 1);

  void TransitionDepthImageLayout(const VkImageLayout old_layout, const VkImageLayout new_layout,
                                  [[maybe_unused]] const VkFormat format);

  void GenerateMipmaps();

  void GenerateMipmaps(VkCommandBuffer command_buffer);

  // void CreateTextureImage(const std::string& image_path);

 private:
  VulkanContext* context_ = nullptr;
  VkFormat image_format_;
  uint32_t mip_levels_ = 1;
  uint32_t array_layers_ = 1;
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkImageTiling tiling_ = VK_IMAGE_TILING_OPTIMAL;
  VkImageUsageFlags usage_ = 0;

 public:
  uint32_t image_width;
  uint32_t image_height;

  VkImage image = VK_NULL_HANDLE;
  VkImageView image_view = VK_NULL_HANDLE;
  VkDeviceMemory image_memory = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace core
