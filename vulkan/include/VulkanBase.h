#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"
#include "VulkanPipelineCache.h"

namespace core {
namespace vulkan {

struct BindingInfo {
  uint32_t binding;
  VkDescriptorType descriptor_type;
  VkShaderStageFlagBits shader_stage;
};

struct DescriptorPoolInfo {
  VkDescriptorType type;
  uint32_t count;
};

class VulkanBase {
 public:
  explicit VulkanBase(VulkanContext* context);
  virtual ~VulkanBase();

  virtual void Init();

  void SetPipelineCache(const VulkanPipelineCache* cache);

  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

 protected:
  virtual std::vector<BindingInfo> GetBindingInfo() const = 0;
  virtual std::vector<VkPushConstantRange> GetPushConstantRanges() const;
  virtual void CreatePipeline() = 0;

 protected:
  std::vector<VkDescriptorSetLayoutBinding> CreateDescriptorLayoutBindings() const;
  std::vector<DescriptorPoolInfo> CreateDescriptorPoolSizes() const;
  VkShaderModule CreateShaderModule(const std::vector<uint32_t>& shader_code) const;
  VkPipelineCache shared_pipeline_cache() const { return shared_pipeline_cache_; }

  void CreateUniformBufferDescriptorSet(const uint32_t binding, const VulkanBuffer& buffer);
  void CreateStorageBufferDescriptorSet(const uint32_t binding, const VulkanBuffer& buffer);
  void CreateCombinedImageSamplerDescriptorSet(const uint32_t binding,
                                               const VkImageView& image_view,
                                               const VkSampler& sampler);

  VulkanContext* context_ = nullptr;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  std::vector<VkDescriptorBufferInfo> buffer_infos_;
  std::vector<VkDescriptorImageInfo> image_infos_;
  std::vector<VkWriteDescriptorSet> writes_;

 private:
  std::mutex write_mutex_;
  VkPipelineCache shared_pipeline_cache_ = VK_NULL_HANDLE;

  int buffer_idx_ = 0;
  int image_idx_ = 0;

  int GetBufferSize(const std::vector<BindingInfo>&& bindings) const {
    int size = 0;
    // TODO: consider other types of buffer
    for (const auto& info : bindings) {
      if (info.descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
          info.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
        size++;
      }
    }
    return size;
  }

  int GetImageSize(const std::vector<BindingInfo>&& bindings) const {
    int size = 0;
    // TODO: consider other types of image
    for (const auto& info : bindings) {
      if (info.descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
          info.descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
          info.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
        size++;
      }
    }
    return size;
  }

  void CheckBufferInfoSize() {
    const auto size = GetBufferSize(GetBindingInfo());
    const auto total_size = GetBindingInfo().size();
    if (buffer_infos_.size() == 0) {
      buffer_infos_.resize(size);
    }
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (writes_.size() == 0) {
      writes_.resize(total_size);
    }
  }

  void CheckImageInfoSize() {
    const auto size = GetImageSize(GetBindingInfo());
    const auto total_size = GetBindingInfo().size();
    if (image_infos_.size() == 0) {
      image_infos_.resize(size);
    }
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (writes_.size() == 0) {
      writes_.resize(total_size);
    }
  }
};

}  // namespace vulkan
}  // namespace core
