#pragma once

#include <optional>
#include <vector>

#include "VulkanBase.h"
#include "VulkanBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"
#include "VulkanRenderPass.h"

namespace core {
namespace vulkan {

struct DynamicRenderingInfo {
  std::vector<VkFormat> color_formats;
  VkFormat depth_format = VK_FORMAT_UNDEFINED;
  VkFormat stencil_format = VK_FORMAT_UNDEFINED;
};

struct DepthBiasInfo {
  float constant_factor = 0.0f;
  float slope_factor = 0.0f;
};

class VulkanRender : public VulkanBase {
 public:
  VulkanRender(VulkanContext* context, VulkanRenderPass* render_pass);
  VulkanRender(VulkanContext* context, const DynamicRenderingInfo& dynamic_rendering_info,
               const VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT);

 protected:
  void CreatePipeline() override;

 private:
  // Derived class can customize these settings
  virtual VkCullModeFlags SetCullMode() const { return VK_CULL_MODE_BACK_BIT; }
  virtual VkFrontFace SetFrontFace() const { return VK_FRONT_FACE_CLOCKWISE; }
  virtual VkBool32 SetDepthTesting() const { return VK_FALSE; }
  virtual VkBool32 SetDepthWriting() const { return VK_FALSE; }
  virtual std::optional<DepthBiasInfo> SetDepthBias() const { return std::nullopt; }
  virtual VkPipelineColorBlendAttachmentState SetColorBlendAttachment() const {
    VkPipelineColorBlendAttachmentState s{};
    s.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    s.blendEnable = VK_FALSE;
    return s;
  }

  // Derived class needs to configure these explicitly
  virtual const std::vector<uint32_t> LoadVertexShader() const = 0;
  virtual const std::vector<uint32_t> LoadFragmentShader() const = 0;
  virtual std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const = 0;
  virtual std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const = 0;

  VulkanRenderPass* render_pass_ = nullptr;
  DynamicRenderingInfo dynamic_rendering_info_{};

  VkSampleCountFlagBits msaa_samples_ = VK_SAMPLE_COUNT_1_BIT;
};

}  // namespace vulkan
}  // namespace core
