#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCompute.h"
#include "GaussianBlurBaseline.slang.spv.h"

namespace core::test {

using vulkan::VulkanBuffer;
using vulkan::VulkanContext;

class BaselineBlur : public vulkan::VulkanCompute {
 public:
  BaselineBlur(VulkanContext* context, VulkanBuffer& source, VulkanBuffer& destination,
               int width, int height)
      : VulkanCompute(context),
        source_(source),
        destination_(destination),
        dimensions_{width, height},
        uniform_(context, sizeof(dimensions_), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        kernel_(context, 9 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {}

  void Init() override {
    VulkanCompute::Init();
    constexpr std::array<float, 9> kWeights = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f,
                                              0.125f, 0.0625f, 0.125f, 0.0625f};
    kernel_.MapData([&kWeights](void* data) {
      std::memcpy(data, kWeights.data(), sizeof(kWeights));
    });
    CreateUniformBufferDescriptorSet(0, uniform_);
    CreateStorageBufferDescriptorSet(1, source_);
    CreateStorageBufferDescriptorSet(2, destination_);
    CreateStorageBufferDescriptorSet(3, kernel_);
    vkUpdateDescriptorSets(context_->logical_device, static_cast<uint32_t>(writes_.size()),
                            writes_.data(), 0, nullptr);
  }

  void Run(VkCommandBuffer command) {
    uniform_.MapData([this](void* data) {
      std::memcpy(data, dimensions_.data(), sizeof(dimensions_));
    });
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                            &descriptor_set_, 0, nullptr);
    vkCmdDispatch(command, static_cast<uint32_t>((dimensions_[0] + 15) / 16),
                   static_cast<uint32_t>((dimensions_[1] + 15) / 16), 1);
  }

 protected:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0, .descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}};
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> code = [] {
      std::vector<uint32_t> words(GaussianBlurBaseline_slang_spv_len / sizeof(uint32_t));
      std::memcpy(words.data(), GaussianBlurBaseline_slang_spv, GaussianBlurBaseline_slang_spv_len);
      return words;
    }();
    return code;
  }

 private:
  VulkanBuffer& source_;
  VulkanBuffer& destination_;
  std::array<int, 2> dimensions_;
  VulkanBuffer uniform_;
  VulkanBuffer kernel_;
};

}
