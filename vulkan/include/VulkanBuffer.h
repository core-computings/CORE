#pragma once

#include <vulkan/vulkan.h>

#include <functional>
#include <iostream>
#include <span>

#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanSync.h"

namespace core {
namespace vulkan {

class VulkanImage;
class VulkanBuffer;

struct BatchBufferCopy {
  const VulkanBuffer* source = nullptr;
  VulkanBuffer* destination = nullptr;
  VkDeviceSize source_offset = 0;
  VkDeviceSize destination_offset = 0;
  VkDeviceSize size = VK_WHOLE_SIZE;
};

class VulkanBuffer {
 public:
  // VulkanBuffer() = delete;  // Buffers must be explicitly initialized
  VulkanBuffer() = default;
  VulkanBuffer(VulkanContext* context, const VkDeviceSize size, const VkBufferUsageFlags usage,
               const VkMemoryPropertyFlags properties);
  ~VulkanBuffer();

  VulkanBuffer& operator=(VulkanBuffer&&);

  void CopyToBuffer(VulkanBuffer& dst_buffer);

  // Execute copies in order with one submission and fence wait. VK_WHOLE_SIZE
  // copies the source's remaining bytes. All buffers must share one context.
  // Validate the entire batch before recording; an empty batch is a no-op.
  // Access to the context's queue must be externally synchronized.
  static void CopyBuffers(std::span<const BatchBufferCopy> copies);

  static void CopyBuffers(VkCommandBuffer command_buffer,
                         std::span<const BatchBufferCopy> copies);

  void CopyToImage(VulkanImage& dst_image, const uint32_t width, const uint32_t height,
                   const uint32_t layers = 1);

  void CopyToImage(VkCommandBuffer command_buffer, VulkanImage& dst_image, uint32_t width,
                   uint32_t height, uint32_t layers = 1);

  void MapData(const std::function<void(void*)>& func);

  VkDeviceSize Size() const { return buffer_size_; }

  VkBuffer buffer = VK_NULL_HANDLE;

  // TODO: keep them public
 private:
  VulkanContext* context_ = nullptr;
  VkDeviceSize buffer_size_ = 0;
  VkMemoryPropertyFlags memory_properties_ = 0;
  VkBufferUsageFlags usage_ = 0;
  VkDeviceMemory buffer_memory_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace core