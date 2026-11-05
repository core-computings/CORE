#pragma once

#include <cstddef>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"

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
  VulkanBase(const VulkanBase&) = delete;
  VulkanBase& operator=(const VulkanBase&) = delete;

  virtual void Init();

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
  size_t GetDescriptorIndex(uint32_t binding, VkDescriptorType type) const;
};

}  // namespace vulkan
}  // namespace core
