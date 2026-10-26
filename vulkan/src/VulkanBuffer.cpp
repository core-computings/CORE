#include "VulkanBuffer.h"

#include <stdexcept>
#include <vector>

#include "VulkanImage.h"

namespace core {
namespace vulkan {

VulkanBuffer::VulkanBuffer(VulkanContext* context, const VkDeviceSize size,
                           const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties)
    : context_(context), buffer_size_(size), memory_properties_(properties), usage_(usage) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = buffer_size_;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK(vkCreateBuffer(context_->logical_device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(context_->logical_device, buffer, &mem_requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
      context_->FindMemoryType(mem_requirements.memoryTypeBits, properties);

  VK_CHECK(vkAllocateMemory(context_->logical_device, &alloc_info, nullptr, &buffer_memory_));

  VK_CHECK(vkBindBufferMemory(context_->logical_device, buffer, buffer_memory_, 0));
}

VulkanBuffer::~VulkanBuffer() {
  if (context_ && buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(context_->logical_device, buffer, nullptr);
  }
  if (context_ && buffer_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(context_->logical_device, buffer_memory_, nullptr);
  }
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& rhs) {
  if (this == &rhs) {
    return *this;
  }

  if (context_ && buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(context_->logical_device, buffer, nullptr);
  }
  if (context_ && buffer_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(context_->logical_device, buffer_memory_, nullptr);
  }

  context_ = rhs.context_;
  buffer = rhs.buffer;
  rhs.buffer = VK_NULL_HANDLE;

  buffer_memory_ = rhs.buffer_memory_;
  rhs.buffer_memory_ = VK_NULL_HANDLE;

  buffer_size_ = rhs.buffer_size_;
  memory_properties_ = rhs.memory_properties_;
  usage_ = rhs.usage_;

  return *this;
}

void VulkanBuffer::CopyToBuffer(VulkanBuffer& dst_buffer) {
  if (buffer_size_ != dst_buffer.Size()) {
    throw std::runtime_error("Buffer sizes do not match for copy");
  }

  const auto command_buffer = VulkanCommandBuffer::BeginOneTimeCommands(context_);
  VkBufferCopy copy_region{};
  copy_region.size = buffer_size_;
  vkCmdCopyBuffer(command_buffer.buffer(), buffer, dst_buffer.buffer, 1, &copy_region);
  command_buffer.EndOneTimeCommands();
}

void VulkanBuffer::CopyBuffers(std::span<const BatchBufferCopy> copies) {
  if (copies.empty()) {
    return;
  }

  if (copies.front().source == nullptr || copies.front().source->context_ == nullptr) {
    throw std::invalid_argument("Buffer copy requires a live source");
  }
  VulkanContext* context = copies.front().source->context_;
  const VulkanCommandBuffer command_buffer = VulkanCommandBuffer::BeginOneTimeCommands(context);
  CopyBuffers(command_buffer.buffer(), copies);
  VulkanFence fence(context);
  fence.Reset();
  command_buffer.Submit(fence.fence);
  VK_CHECK(vkWaitForFences(context->logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
}

void VulkanBuffer::CopyBuffers(VkCommandBuffer command_buffer,
                              std::span<const BatchBufferCopy> copies) {
  if (copies.empty()) {
    return;
  }
  if (command_buffer == VK_NULL_HANDLE) {
    throw std::invalid_argument("Buffer copy requires a command buffer");
  }

  VulkanContext* context = nullptr;
  std::vector<VkBufferCopy> regions;
  regions.reserve(copies.size());
  for (const BatchBufferCopy& copy : copies) {
    if (copy.source == nullptr || copy.destination == nullptr) {
      throw std::invalid_argument("Buffer copy requires source and destination");
    }
    const VulkanBuffer& source = *copy.source;
    const VulkanBuffer& destination = *copy.destination;
    if (context == nullptr) {
      context = source.context_;
    }
    if (context == nullptr || source.context_ != context || destination.context_ != context ||
        source.buffer == VK_NULL_HANDLE || destination.buffer == VK_NULL_HANDLE) {
      throw std::invalid_argument("Buffer copies require live buffers from one context");
    }
    if ((source.usage_ & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0 ||
        (destination.usage_ & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0) {
      throw std::invalid_argument("Buffer copy requires transfer usage flags");
    }
    if (copy.source_offset >= source.buffer_size_ ||
        copy.destination_offset >= destination.buffer_size_) {
      throw std::out_of_range("Buffer copy offset is outside the buffer");
    }
    const VkDeviceSize size = copy.size == VK_WHOLE_SIZE
                                  ? source.buffer_size_ - copy.source_offset
                                  : copy.size;
    if (size == 0 || size > source.buffer_size_ - copy.source_offset ||
        size > destination.buffer_size_ - copy.destination_offset) {
      throw std::out_of_range("Buffer copy size is outside the buffer");
    }
    if (source.buffer == destination.buffer &&
        copy.source_offset < copy.destination_offset + size &&
        copy.destination_offset < copy.source_offset + size) {
      throw std::invalid_argument("Buffer copy source and destination overlap");
    }
    regions.push_back(VkBufferCopy{
        .srcOffset = copy.source_offset,
        .dstOffset = copy.destination_offset,
        .size = size,
    });
  }

  const VkMemoryBarrier before{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
  };
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);

  const VkMemoryBarrier between{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
  };
  for (size_t index = 0; index < copies.size(); ++index) {
    if (index != 0) {
      // Preserve array order, including read-after-write and overlapping writes.
      vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &between, 0, nullptr, 0, nullptr);
    }
    vkCmdCopyBuffer(command_buffer, copies[index].source->buffer,
                    copies[index].destination->buffer, 1, &regions[index]);
  }

  const VkMemoryBarrier after{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_HOST_READ_BIT,
  };
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                       &after, 0, nullptr, 0, nullptr);
}

void VulkanBuffer::CopyToImage(VulkanImage& dst_image, const uint32_t width, const uint32_t height,
                               const uint32_t layers) {
  const VulkanCommandBuffer command_buffer = VulkanCommandBuffer::BeginOneTimeCommands(context_);
  CopyToImage(command_buffer.buffer(), dst_image, width, height, layers);
  command_buffer.EndOneTimeCommands();
}

void VulkanBuffer::CopyToImage(VkCommandBuffer command_buffer, VulkanImage& dst_image,
                             uint32_t width, uint32_t height, uint32_t layers) {
  if (layers == 0) {
    throw std::invalid_argument("Image copy requires at least one layer");
  }

  std::vector<VkBufferImageCopy> regions(layers);
  const VkDeviceSize layer_size = buffer_size_ / layers;

  for (uint32_t i = 0; i < layers; ++i) {
    VkBufferImageCopy region{};
    region.bufferOffset = layer_size * i;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = i;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {.x = 0, .y = 0, .z = 0};
    region.imageExtent = {.width = width, .height = height, .depth = 1};

    regions[i] = region;
  }

  vkCmdCopyBufferToImage(command_buffer, buffer, dst_image.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layers, regions.data());
}

void VulkanBuffer::MapData(const std::function<void(void*)>& func) {
  const auto staging_buffer_property =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if ((memory_properties_ & staging_buffer_property) != staging_buffer_property) {
    throw std::runtime_error("Buffer is not mappable");
  }
  void* data;
  vkMapMemory(context_->logical_device, buffer_memory_, 0, buffer_size_, 0, &data);
  func(data);
  vkUnmapMemory(context_->logical_device, buffer_memory_);
}

}  // namespace vulkan
}  // namespace core