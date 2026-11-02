#include "VulkanQueryPool.h"

#include <stdexcept>

namespace core {
namespace vulkan {

VulkanQueryPool::VulkanQueryPool(VulkanContext* context, VkQueryType query_type)
    : VulkanQueryPool(context, query_type, 2) {}

VulkanQueryPool::VulkanQueryPool(VulkanContext* context, VkQueryType query_type,
                               uint32_t query_count)
    : context_(context), query_count_(query_count) {
  if (query_type != VK_QUERY_TYPE_TIMESTAMP) {
    throw std::runtime_error("Query type is not supported");
  }
  if (query_count == 0) {
    throw std::invalid_argument("Timestamp query count must be positive");
  }
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device, &family_count,
                                          families.data());
  const QueueFamilyIndices indices = context_->GetQueueFamilyIndices();
  const uint32_t family = context_->queue_family_type() == QueueFamilyType::Compute
                              ? indices.compute_family.value()
                              : indices.graphics_family.value();
  const uint32_t valid_bits = families.at(family).timestampValidBits;
  if (valid_bits == 0) {
    throw std::runtime_error("Selected queue family does not support timestamps");
  }
  timestamp_mask_ = UINT64_MAX >> (64U - valid_bits);
  timestamps_.resize(query_count_);
  const VkQueryPoolCreateInfo pool_info{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queryType = query_type,
      .queryCount = query_count_,
      .pipelineStatistics = 0,
  };
  VK_CHECK(vkCreateQueryPool(context_->logical_device, &pool_info, nullptr, &query_pool_));
}

VulkanQueryPool::~VulkanQueryPool() {
  vkDestroyQueryPool(context_->logical_device, query_pool_, nullptr);
}

void VulkanQueryPool::Reset(VkCommandBuffer command) {
  if (command == VK_NULL_HANDLE) {
    throw std::invalid_argument("Timestamp reset requires a command buffer");
  }
  results_available_ = false;
  vkCmdResetQueryPool(command, query_pool_, 0, query_count_);
}

void VulkanQueryPool::Query(VkCommandBuffer command, uint32_t query,
                            VkPipelineStageFlagBits stage) {
  if (command == VK_NULL_HANDLE || query >= query_count_) {
    throw std::invalid_argument("Timestamp query requires a command buffer and valid index");
  }
  results_available_ = false;
  vkCmdWriteTimestamp(command, stage, query_pool_, query);
}

void VulkanQueryPool::GetQueryResults() {
  results_available_ = false;
  VK_CHECK(vkGetQueryPoolResults(context_->logical_device, query_pool_, 0, query_count_,
                                timestamps_.size() * sizeof(uint64_t), timestamps_.data(),
                                sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  results_available_ = true;
}

double VulkanQueryPool::GetElapsedMilliseconds(uint32_t first, uint32_t last) const {
  if (first >= query_count_ || last >= query_count_ || first > last) {
    throw std::invalid_argument("Timestamp interval requires ordered valid indices");
  }
  if (!results_available_) {
    throw std::logic_error("Timestamp results have not been retrieved for this recording");
  }
  const uint64_t ticks = (timestamps_[last] - timestamps_[first]) & timestamp_mask_;
  return static_cast<double>(ticks) * static_cast<double>(context_->timestamp_period) / 1000000.0;
}

}
}
