#include "VulkanBase.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace core {
namespace vulkan {

// Descriptor visualization: https://chuzcjoe.github.io/CGV/cgv-vulkan-visualization/

VulkanBase::VulkanBase(VulkanContext* context) : context_(context) {}

VulkanBase::~VulkanBase() {
  vkDestroyPipeline(context_->logical_device, pipeline, nullptr);
  vkDestroyPipelineLayout(context_->logical_device, pipeline_layout, nullptr);
  vkDestroyDescriptorPool(context_->logical_device, descriptor_pool_, nullptr);
  vkDestroyDescriptorSetLayout(context_->logical_device, descriptor_set_layout_, nullptr);
}

void VulkanBase::Init() {
  const std::vector<VkPushConstantRange> push_constant_ranges =
      GetPushConstantRanges();
  if (!push_constant_ranges.empty()) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context_->physical_device, &properties);
    const uint32_t limit = properties.limits.maxPushConstantsSize;
    VkShaderStageFlags used_stages = 0;
    for (const VkPushConstantRange& range : push_constant_ranges) {
      if (range.stageFlags == 0 || (range.stageFlags & used_stages) != 0 ||
          range.size == 0 || range.offset % 4 != 0 || range.size % 4 != 0 ||
          range.offset >= limit || range.size > limit - range.offset) {
        throw std::invalid_argument("Invalid push constant range");
      }
      used_stages |= range.stageFlags;
    }
  }

  // 1. Create descriptor set layout
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  const auto binding_info = CreateDescriptorLayoutBindings();
  std::unordered_set<uint32_t> declared_bindings;
  for (const VkDescriptorSetLayoutBinding& binding : binding_info) {
    if (!declared_bindings.insert(binding.binding).second) {
      throw std::invalid_argument("Duplicate descriptor binding");
    }
  }
  layout_info.bindingCount = static_cast<uint32_t>(binding_info.size());
  layout_info.pBindings = binding_info.data();
  VK_CHECK(vkCreateDescriptorSetLayout(context_->logical_device, &layout_info, nullptr,
                                       &descriptor_set_layout_));

  // 2. Create descriptor sets
  const auto pool_info = CreateDescriptorPoolSizes();
  std::vector<VkDescriptorPoolSize> pool_sizes;
  for (const auto& info : pool_info) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = info.type;
    pool_size.descriptorCount = info.count;
    pool_sizes.emplace_back(pool_size);
  }
  VkDescriptorPoolCreateInfo pool_create_info{};
  pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  pool_create_info.pPoolSizes = pool_sizes.data();
  pool_create_info.maxSets = 1;
  VK_CHECK(vkCreateDescriptorPool(context_->logical_device, &pool_create_info, nullptr,
                                  &descriptor_pool_));

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout_;
  VK_CHECK(vkAllocateDescriptorSets(context_->logical_device, &alloc_info, &descriptor_set_));

  buffer_infos_.resize(binding_info.size());
  image_infos_.resize(binding_info.size());
  writes_.resize(binding_info.size());
  for (size_t index = 0; index < binding_info.size(); ++index) {
    VkWriteDescriptorSet& write = writes_[index];
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = binding_info[index].binding;
    write.descriptorType = binding_info[index].descriptorType;
    write.descriptorCount = 1;
  }

  // 3. Create pipeline layout
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
  pipeline_layout_info.pushConstantRangeCount =
      static_cast<uint32_t>(push_constant_ranges.size());
  pipeline_layout_info.pPushConstantRanges = push_constant_ranges.data();
  VK_CHECK(vkCreatePipelineLayout(context_->logical_device, &pipeline_layout_info, nullptr,
                                  &pipeline_layout));

  // 4. Create pipeline
  CreatePipeline();
}

std::vector<VkPushConstantRange> VulkanBase::GetPushConstantRanges() const {
  return {};
}

std::vector<VkDescriptorSetLayoutBinding> VulkanBase::CreateDescriptorLayoutBindings() const {
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  const auto binding_info = GetBindingInfo();
  for (const auto& info : binding_info) {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = info.binding;
    binding.descriptorType = info.descriptor_type;
    binding.descriptorCount = 1;  // Assuming one descriptor per binding
    binding.stageFlags = info.shader_stage;
    binding.pImmutableSamplers = nullptr;  // Optional
    bindings.emplace_back(binding);
  }
  return bindings;
}

std::vector<DescriptorPoolInfo> VulkanBase::CreateDescriptorPoolSizes() const {
  std::vector<DescriptorPoolInfo> pool_info;
  const auto binding_info = GetBindingInfo();
  std::unordered_map<VkDescriptorType, uint32_t> type_count;
  for (const auto& info : binding_info) {
    type_count[info.descriptor_type]++;
  }
  for (const auto& [type, count] : type_count) {
    DescriptorPoolInfo info{type, count};
    pool_info.emplace_back(info);
  }
  return pool_info;
}

VkShaderModule VulkanBase::CreateShaderModule(const std::vector<uint32_t>& shader_code) const {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = shader_code.size() * sizeof(uint32_t);
  create_info.pCode = shader_code.data();

  VkShaderModule shader_module;
  VK_CHECK(vkCreateShaderModule(context_->logical_device, &create_info, nullptr, &shader_module));
  return shader_module;
}

size_t VulkanBase::GetDescriptorIndex(uint32_t binding, VkDescriptorType type) const {
  if (descriptor_set_ == VK_NULL_HANDLE) {
    throw std::logic_error(
        "Initialize the pipeline before preparing descriptors");
  }
  const auto write =
      std::find_if(writes_.begin(), writes_.end(),
                   [binding](const VkWriteDescriptorSet& candidate) {
                     return candidate.dstBinding == binding;
                   });
  if (write == writes_.end() || write->descriptorType != type) {
    throw std::invalid_argument(
        "Descriptor binding is undeclared or has a different type");
  }
  return static_cast<size_t>(write - writes_.begin());
}

void VulkanBase::CreateUniformBufferDescriptorSet(const uint32_t binding,
                                                  const VulkanBuffer& buffer) {
  const size_t index =
      GetDescriptorIndex(binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  buffer_infos_[index] = {
      .buffer = buffer.buffer, .offset = 0, .range = buffer.Size()};
  writes_[index].pBufferInfo = &buffer_infos_[index];
}

void VulkanBase::CreateStorageBufferDescriptorSet(const uint32_t binding,
                                                  const VulkanBuffer& buffer) {
  const size_t index =
      GetDescriptorIndex(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  buffer_infos_[index] = {
      .buffer = buffer.buffer, .offset = 0, .range = buffer.Size()};
  writes_[index].pBufferInfo = &buffer_infos_[index];
}

void VulkanBase::CreateCombinedImageSamplerDescriptorSet(
    const uint32_t binding, const VkImageView& image_view,
    const VkSampler& sampler) {
  const size_t index =
      GetDescriptorIndex(binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  image_infos_[index] = {
      .sampler = sampler,
      .imageView = image_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  writes_[index].pImageInfo = &image_infos_[index];
}

}  // namespace vulkan
}  // namespace core
