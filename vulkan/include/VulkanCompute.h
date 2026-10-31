#pragma once

#include <cstddef>
#include <vector>

#include "VulkanBase.h"
#include "VulkanBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"

namespace core {
namespace vulkan {

struct ComputeSpecialization {
  std::vector<VkSpecializationMapEntry> entries;
  std::vector<std::byte> data;
};

class VulkanCompute : public VulkanBase {
 public:
  explicit VulkanCompute(VulkanContext* context);

  ~VulkanCompute();

 protected:
  void CreatePipeline() override;
  virtual ComputeSpecialization GetSpecialization() const;

  // Derived pipelines need to implement this if pipeline cache is used
  virtual const std::string GetPipelineCache() const;

  void SavePipelineCache(const std::string& cache) const;

 private:
  virtual const std::vector<uint32_t>& LoadShaderCode() const = 0;

  void CreatePipelineCache(const std::string& cache);

  static std::vector<char> LoadPipelineCache(const std::string& cache);

 protected:
  VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace core