#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "VulkanContext.h"

namespace core {
namespace vulkan {

class VulkanPipelineCache {
 public:
  static constexpr std::uintmax_t kMaxDataSize = std::uintmax_t{64} << 20U;

  VulkanPipelineCache(VulkanContext* context, std::filesystem::path path);
  ~VulkanPipelineCache();

  VulkanPipelineCache(const VulkanPipelineCache&) = delete;
  VulkanPipelineCache& operator=(const VulkanPipelineCache&) = delete;

  VkPipelineCache cache() const { return cache_; }
  const std::filesystem::path& path() const { return path_; }
  bool restored() const { return restored_; }

  bool Save() const;

  static bool IsCompatible(const std::vector<std::byte>& data,
                           const VkPhysicalDeviceProperties& properties);
  static bool Write(VkDevice device, VkPipelineCache cache, const std::filesystem::path& path);

 private:
  VulkanContext* context_ = nullptr;
  std::filesystem::path path_;
  VkPipelineCache cache_ = VK_NULL_HANDLE;
  bool restored_ = false;
};

}
}
