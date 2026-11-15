#pragma once

#ifdef __APPLE__
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

#include <vulkan/vulkan.h>

#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "VulkanUtils.h"

namespace core {
namespace vulkan {

struct QueueFamilyIndices {
  std::optional<uint32_t> compute_family;
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;

  bool is_complete() {
    return compute_family.has_value() ||
           (graphics_family.has_value() && present_family.has_value());
  }
};

struct Float16Support {
  bool arithmetic = false;
  bool storage_buffer = false;
};

/*
Vulkan context manages:
1. instance
2. physical device
3. logical device
4. queue families
5. queues
6. debug messenger
*/

class VulkanContext {
 public:
  VulkanContext(const bool enable_validation_layers = false,
                const QueueFamilyType queue_family_type = QueueFamilyType::Compute,
                const VkSurfaceKHR surface = VK_NULL_HANDLE);
  ~VulkanContext();

  void Init(VkSurfaceKHR surface = VK_NULL_HANDLE);

  uint32_t FindMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties) const;

  QueueFamilyIndices GetQueueFamilyIndices() const { return queue_family_indices_; }

  VkQueue compute_queue() const { return compute_queue_; }
  VkQueue graphics_queue() const { return graphics_queue_; }
  VkQueue present_queue() const { return present_queue_; }
  QueueFamilyType queue_family_type() const { return queue_family_type_; }
  const Float16Support& float16_support() const { return float16_support_; }

  VkInstance instance = VK_NULL_HANDLE;
  VkDevice logical_device;
  VkPhysicalDevice physical_device;
  float timestamp_period;

 private:
  bool enable_validation_layers_;
  QueueFamilyType queue_family_type_;
  QueueFamilyIndices queue_family_indices_;
  VkQueue compute_queue_ = VK_NULL_HANDLE;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  Float16Support float16_support_;

  VkDebugUtilsMessengerEXT debug_messenger_;

  bool CheckValidationLayerSupport();
  void CreateInstance(const bool enable_validation_layers);
  void PickPhysicalDevice(const QueueFamilyType queue_family_type);
  void FindQueueFamilies(VkPhysicalDevice device, const QueueFamilyType queue_family_type);
  void CreateLogicalDevice(const float queuePriority = 1.0f);
  std::vector<const char*> GetRequiredDeviceExtensions() const;
  void SetupDebugMessenger();
};
}  // namespace vulkan
}  // namespace core
