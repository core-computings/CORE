#include "VulkanCompute.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace core {
namespace vulkan {

VulkanCompute::VulkanCompute(VulkanContext* context) : VulkanBase(context) {}

VulkanCompute::~VulkanCompute() = default;

void VulkanCompute::CreatePipeline() {
  const ComputeSpecialization specialization = GetSpecialization();
  if (specialization.entries.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Too many compute specialization entries");
  }
  std::unordered_set<uint32_t> constant_ids;
  for (const VkSpecializationMapEntry& entry : specialization.entries) {
    if (entry.size == 0 || entry.offset > specialization.data.size() ||
        entry.size > specialization.data.size() - entry.offset ||
        !constant_ids.insert(entry.constantID).second) {
      throw std::invalid_argument("Invalid compute specialization entry");
    }
  }
  const VkSpecializationInfo specialization_info{
      .mapEntryCount = static_cast<uint32_t>(specialization.entries.size()),
      .pMapEntries = specialization.entries.data(),
      .dataSize = specialization.data.size(),
      .pData = specialization.data.data()};

  const std::vector<uint32_t>& shader_code = LoadShaderCode();
  const VkShaderModule shader_module = CreateShaderModule(shader_code);
  struct ShaderModuleScope {
    VkDevice device;
    VkShaderModule module;
    ~ShaderModuleScope() { vkDestroyShaderModule(device, module, nullptr); }
  } shader_scope{.device = context_->logical_device, .module = shader_module};

  if (shared_pipeline_cache() != VK_NULL_HANDLE) {
    owned_pipeline_cache_.reset();
    pipeline_cache_ = shared_pipeline_cache();
  } else {
    owned_pipeline_cache_ = std::make_unique<VulkanPipelineCache>(context_, GetPipelineCache());
    pipeline_cache_ = owned_pipeline_cache_->cache();
  }

  VkPipelineShaderStageCreateInfo shader_stage_info{};
  shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage_info.module = shader_module;
  shader_stage_info.pName = "main";
  shader_stage_info.pSpecializationInfo =
      specialization.entries.empty() ? nullptr : &specialization_info;

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = shader_stage_info;
  pipeline_info.layout = pipeline_layout;

  VK_CHECK(vkCreateComputePipelines(context_->logical_device, pipeline_cache_, 1, &pipeline_info,
                                    nullptr, &pipeline));
}

ComputeSpecialization VulkanCompute::GetSpecialization() const { return {}; }

const std::string VulkanCompute::GetPipelineCache() const { return ""; }

void VulkanCompute::SavePipelineCache(const std::string& cache) const {
  VulkanPipelineCache::Write(context_->logical_device, pipeline_cache_, cache);
}

}  // namespace vulkan
}  // namespace core