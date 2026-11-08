#include "VulkanPipelineCache.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <random>
#include <string>
#include <system_error>
#include <utility>

namespace core {
namespace vulkan {

namespace {

constexpr size_t kHeaderSize = 32;

uint32_t ReadLittleEndian32(const std::vector<std::byte>& data, size_t offset) {
  uint32_t value = 0;
  for (size_t index = 0; index < sizeof(uint32_t); ++index) {
    value |= static_cast<uint32_t>(std::to_integer<uint8_t>(data[offset + index]))
             << (8U * static_cast<uint32_t>(index));
  }
  return value;
}

std::vector<std::byte> ReadCompatibleFile(const std::filesystem::path& path,
                                          const VkPhysicalDeviceProperties& properties) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size < kHeaderSize || size > VulkanPipelineCache::kMaxDataSize) {
    return {};
  }
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  std::vector<std::byte> data(kHeaderSize);
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(kHeaderSize));
  if (!file || !VulkanPipelineCache::IsCompatible(data, properties)) {
    return {};
  }
  data.resize(static_cast<size_t>(size));
  file.read(reinterpret_cast<char*>(data.data() + kHeaderSize),
            static_cast<std::streamsize>(size - kHeaderSize));
  if (!file || file.peek() != std::ifstream::traits_type::eof()) {
    return {};
  }
  return data;
}

}

VulkanPipelineCache::VulkanPipelineCache(VulkanContext* context, std::filesystem::path path)
    : context_(context), path_(std::move(path)) {
  std::vector<std::byte> data;
  if (!path_.empty()) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context_->physical_device, &properties);
    data = ReadCompatibleFile(path_, properties);
  }
  restored_ = !data.empty();

  const VkPipelineCacheCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .initialDataSize = data.size(),
      .pInitialData = restored_ ? data.data() : nullptr,
  };
  VK_CHECK(vkCreatePipelineCache(context_->logical_device, &create_info, nullptr, &cache_));
}

VulkanPipelineCache::~VulkanPipelineCache() {
  vkDestroyPipelineCache(context_->logical_device, cache_, nullptr);
}

bool VulkanPipelineCache::Save() const { return Write(context_->logical_device, cache_, path_); }

bool VulkanPipelineCache::IsCompatible(const std::vector<std::byte>& data,
                                       const VkPhysicalDeviceProperties& properties) {
  if (data.size() < kHeaderSize) {
    return false;
  }
  for (size_t index = 0; index < VK_UUID_SIZE; ++index) {
    if (std::to_integer<uint8_t>(data[16 + index]) != properties.pipelineCacheUUID[index]) {
      return false;
    }
  }
  return ReadLittleEndian32(data, 0) == static_cast<uint32_t>(kHeaderSize) &&
         ReadLittleEndian32(data, 4) ==
             static_cast<uint32_t>(VK_PIPELINE_CACHE_HEADER_VERSION_ONE) &&
         ReadLittleEndian32(data, 8) == properties.vendorID &&
         ReadLittleEndian32(data, 12) == properties.deviceID;
}

bool VulkanPipelineCache::Write(VkDevice device, VkPipelineCache cache,
                                const std::filesystem::path& path) {
  if (cache == VK_NULL_HANDLE || path.empty()) {
    return false;
  }

  std::vector<std::byte> data;
  VkResult result = VK_INCOMPLETE;
  while (result == VK_INCOMPLETE) {
    size_t size = 0;
    VK_CHECK(vkGetPipelineCacheData(device, cache, &size, nullptr));
    if (static_cast<std::uintmax_t>(size) > kMaxDataSize) {
      return false;
    }
    data.resize(size);
    result = vkGetPipelineCacheData(device, cache, &size, data.data());
    if (result != VK_INCOMPLETE) {
      VK_CHECK(result);
      data.resize(size);
    }
  }
  if (data.empty()) {
    return false;
  }

  std::random_device random;
  const uint64_t suffix =
      (static_cast<uint64_t>(random()) << 32U) | static_cast<uint64_t>(random());
  std::filesystem::path temp_path = path;
  temp_path += ".tmp" + std::to_string(suffix);

  std::error_code error;
  {
    std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    file.close();
    if (!file) {
      std::filesystem::remove(temp_path, error);
      return false;
    }
  }
  std::filesystem::rename(temp_path, path, error);
  if (error) {
    std::filesystem::remove(temp_path, error);
    return false;
  }
  return true;
}

}
}
