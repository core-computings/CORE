#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D16_UNORM;
constexpr VkSampleCountFlagBits kSamples = VK_SAMPLE_COUNT_4_BIT;
constexpr uint32_t kExtent = 32;
constexpr uint32_t kRed = 0xff0000ffU;
constexpr uint32_t kGreen = 0xff00ff00U;
constexpr uint32_t kBlue = 0xffff0000U;

struct DrawConstants {
  float color[4];
  float depth;
};

struct GraphicsPipeline {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule vertex = VK_NULL_HANDLE;
  VkShaderModule fragment = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  ~GraphicsPipeline() {
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyShaderModule(device, fragment, nullptr);
    vkDestroyShaderModule(device, vertex, nullptr);
  }
};

class VulkanTransientAttachmentTest : public testing::Test {
 protected:
  void SetUp() override { context_.Init(); }

  bool SupportsGraphics() const {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context_.physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(context_.physical_device, &count, families.data());
    const uint32_t family = context_.GetQueueFamilyIndices().compute_family.value();
    return (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
  }

  bool HasLazyMemoryType(const vulkan::VulkanImage& image) const {
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(context_.logical_device, image.image, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(context_.physical_device, &memory);
    const VkMemoryPropertyFlags lazy =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1U << i)) != 0U &&
          (memory.memoryTypes[i].propertyFlags & lazy) == lazy) {
        return true;
      }
    }
    return false;
  }

  vulkan::VulkanImage CreateColor(VkImageUsageFlags usage) {
    return vulkan::VulkanImage(&context_, kExtent, kExtent, kColorFormat, usage,
                               VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               VK_IMAGE_TILING_OPTIMAL, 1, kSamples);
  }

  vulkan::VulkanImage CreateDepth(VkImageUsageFlags usage) {
    return vulkan::VulkanImage(&context_, kExtent, kExtent, kDepthFormat, usage,
                               VK_IMAGE_ASPECT_DEPTH_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               VK_IMAGE_TILING_OPTIMAL, 1, kSamples);
  }

  void ExpectMemory(const vulkan::VulkanImage& image, bool transient) {
    const VkMemoryPropertyFlags properties = image.memory_properties();
    const bool lazy = (properties & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0U;
    EXPECT_NE(properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U);
    EXPECT_EQ(lazy, transient && HasLazyMemoryType(image));
    VkDeviceSize committed = 0;
    if (lazy) {
      vkGetDeviceMemoryCommitment(context_.logical_device, image.image_memory, &committed);
    }
    std::printf("transient=%d lazy=%d properties=0x%x committed=%llu\n", transient ? 1 : 0,
                lazy ? 1 : 0, static_cast<unsigned>(properties),
                static_cast<unsigned long long>(committed));
  }

  void CreatePipeline(GraphicsPipeline& pipeline) {
    static const std::vector<uint32_t> vertex_code =
#include "TransientAttachment.vert.spv"
        ;
    static const std::vector<uint32_t> fragment_code =
#include "TransientAttachment.frag.spv"
        ;
    pipeline.device = context_.logical_device;
    const VkShaderModuleCreateInfo vertex_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = vertex_code.size() * sizeof(uint32_t),
        .pCode = vertex_code.data(),
    };
    ASSERT_EQ(vkCreateShaderModule(pipeline.device, &vertex_info, nullptr, &pipeline.vertex),
              VK_SUCCESS);
    const VkShaderModuleCreateInfo fragment_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = fragment_code.size() * sizeof(uint32_t),
        .pCode = fragment_code.data(),
    };
    ASSERT_EQ(vkCreateShaderModule(pipeline.device, &fragment_info, nullptr, &pipeline.fragment),
              VK_SUCCESS);

    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(DrawConstants),
    };
    const VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    };
    ASSERT_EQ(vkCreatePipelineLayout(pipeline.device, &layout_info, nullptr, &pipeline.layout),
              VK_SUCCESS);

    const VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = pipeline.vertex,
         .pName = "main",
         .pSpecializationInfo = nullptr},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = pipeline.fragment,
         .pName = "main",
         .pSpecializationInfo = nullptr},
    };
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = kSamples;
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                              VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &kColorFormat,
        .depthAttachmentFormat = kDepthFormat,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &rendering;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = pipeline.layout;
    ASSERT_EQ(vkCreateGraphicsPipelines(pipeline.device, VK_NULL_HANDLE, 1, &pipeline_info,
                                        nullptr, &pipeline.pipeline),
              VK_SUCCESS);
  }

  void Draw(VkCommandBuffer command, const GraphicsPipeline& pipeline, uint32_t color,
            float depth, uint32_t width) {
    const DrawConstants constants{
        .color = {static_cast<float>(color & 0xffU) / 255.0f,
                  static_cast<float>((color >> 8U) & 0xffU) / 255.0f,
                  static_cast<float>((color >> 16U) & 0xffU) / 255.0f, 1.0f},
        .depth = depth,
    };
    const VkRect2D scissor{.offset = {.x = 0, .y = 0},
                           .extent = {.width = width, .height = kExtent}};
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdPushConstants(command, pipeline.layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(DrawConstants), &constants);
    vkCmdDraw(command, 3, 1, 0, 0);
  }

  template <typename Record>
  std::vector<uint32_t> RenderFrame(const GraphicsPipeline& pipeline,
                                    const vulkan::VulkanDynamicRendering& rendering,
                                    vulkan::VulkanImage& color, vulkan::VulkanImage& depth,
                                    vulkan::VulkanImage& resolve, Record record) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(kExtent) * kExtent * sizeof(uint32_t);
    vulkan::VulkanBuffer readback(
        &context_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const vulkan::VulkanCommandBuffer command =
        vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context_);
    resolve.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kColorFormat);
    const VkClearValue color_clear{.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
    const VkClearValue depth_clear{.depthStencil = {.depth = 1.0f, .stencil = 0}};
    const VkExtent2D extent{.width = kExtent, .height = kExtent};
    rendering.BeginDynamicRendering(command.buffer(), color.image_view, resolve.image_view, extent,
                                    color_clear, depth.image_view, depth_clear);
    vkCmdBindPipeline(command.buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    const VkViewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(kExtent),
                              .height = static_cast<float>(kExtent),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};
    vkCmdSetViewport(command.buffer(), 0, 1, &viewport);
    record(command.buffer());
    rendering.EndDynamicRendering(command.buffer());

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = resolve.image;
    barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy region{};
    region.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .mipLevel = 0,
                               .baseArrayLayer = 0,
                               .layerCount = 1};
    region.imageExtent = {.width = kExtent, .height = kExtent, .depth = 1};
    vkCmdCopyImageToBuffer(command.buffer(), resolve.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &region);
    const VkMemoryBarrier host_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);

    vulkan::VulkanFence fence(&context_);
    fence.Reset();
    command.Submit(fence.fence);
    EXPECT_EQ(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX),
              VK_SUCCESS);
    std::vector<uint32_t> pixels(static_cast<size_t>(kExtent) * kExtent);
    readback.MapData([&](void* data) { std::memcpy(pixels.data(), data, bytes); });
    return pixels;
  }

  vulkan::VulkanContext context_{true};
};

TEST_F(VulkanTransientAttachmentTest, TransientAttachmentsPreferLazilyAllocatedMemory) {
  vulkan::VulkanImage color =
      CreateColor(VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  vulkan::VulkanImage depth = CreateDepth(VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  ExpectMemory(color, true);
  ExpectMemory(depth, true);

  const VkMemoryPropertyFlags properties = color.memory_properties();
  vulkan::VulkanImage moved;
  moved = std::move(color);
  EXPECT_EQ(moved.memory_properties(), properties);
}

TEST_F(VulkanTransientAttachmentTest, PersistentImagesKeepRequestedMemory) {
  vulkan::VulkanImage color =
      CreateColor(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  vulkan::VulkanImage depth = CreateDepth(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  ExpectMemory(color, false);
  ExpectMemory(depth, false);
}

TEST_F(VulkanTransientAttachmentTest, RendersThroughTransientMultisampledAttachments) {
  if (!SupportsGraphics()) {
    GTEST_SKIP() << "Selected queue family does not support graphics";
  }
  vulkan::VulkanImage color =
      CreateColor(VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  vulkan::VulkanImage depth = CreateDepth(VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  vulkan::VulkanImage resolve(
      &context_, kExtent, kExtent, kColorFormat,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  color.TransitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              kColorFormat);
  depth.TransitionDepthImageLayout(VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, kDepthFormat);
  GraphicsPipeline pipeline;
  ASSERT_NO_FATAL_FAILURE(CreatePipeline(pipeline));
  const vulkan::VulkanDynamicRendering rendering(&context_);

  const std::vector<uint32_t> first =
      RenderFrame(pipeline, rendering, color, depth, resolve, [&](VkCommandBuffer command) {
        Draw(command, pipeline, kRed, 0.25f, kExtent);
        Draw(command, pipeline, kGreen, 0.75f, kExtent);
        Draw(command, pipeline, kBlue, 0.1f, kExtent / 2);
      });
  for (uint32_t y = 0; y < kExtent; ++y) {
    for (uint32_t x = 0; x < kExtent; ++x) {
      ASSERT_EQ(first[static_cast<size_t>(y) * kExtent + x], x < kExtent / 2 ? kBlue : kRed)
          << "x=" << x << " y=" << y;
    }
  }

  const std::vector<uint32_t> second =
      RenderFrame(pipeline, rendering, color, depth, resolve, [&](VkCommandBuffer command) {
        Draw(command, pipeline, kGreen, 0.9f, kExtent);
      });
  for (size_t index = 0; index < second.size(); ++index) {
    ASSERT_EQ(second[index], kGreen) << "index=" << index;
  }
}

}
}
}
