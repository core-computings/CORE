#pragma once

#include "VulkanContext.h"

namespace core {
namespace vulkan {

class VulkanQueryPool {
 public:
  VulkanQueryPool(VulkanContext* context, VkQueryType query_type);
  VulkanQueryPool(VulkanContext* context, VkQueryType query_type, uint32_t query_count);
  ~VulkanQueryPool();

  void Reset(const VkCommandBuffer command);
  void Query(const VkCommandBuffer command, uint32_t query,
             VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  void GetQueryResults();

  double GetElapsedMilliseconds(uint32_t first, uint32_t last) const;

  std::vector<uint64_t> GetTimeStamps() const { return timestamps_; }

 private:
  VulkanContext* context_ = nullptr;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  uint32_t query_count_ = 0;
  uint64_t timestamp_mask_ = UINT64_MAX;
  bool results_available_ = false;

  std::vector<uint64_t> timestamps_;
};
}  // namespace vulkan
}  // namespace core