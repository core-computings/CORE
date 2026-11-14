#include "VulkanRender.h"

#include <cmath>
#include <stdexcept>

namespace core {
namespace vulkan {

VulkanRender::VulkanRender(VulkanContext* context, VulkanRenderPass* render_pass)
    : VulkanBase(context), render_pass_(render_pass) {}

VulkanRender::VulkanRender(VulkanContext* context,
                           const DynamicRenderingInfo& dynamic_rendering_info,
                           const VkSampleCountFlagBits msaa_samples)
    : VulkanBase(context),
      render_pass_(nullptr),
      dynamic_rendering_info_(dynamic_rendering_info),
      msaa_samples_(msaa_samples) {}

void VulkanRender::CreatePipeline() {
  const bool has_color_attachment =
      render_pass_ != nullptr || !dynamic_rendering_info_.color_formats.empty();
  if (render_pass_ == nullptr) {
    if (dynamic_rendering_info_.color_formats.size() > 1) {
      throw std::runtime_error("Dynamic rendering currently supports at most one color format");
    }
    if (!has_color_attachment && dynamic_rendering_info_.depth_format == VK_FORMAT_UNDEFINED &&
        dynamic_rendering_info_.stencil_format == VK_FORMAT_UNDEFINED) {
      throw std::invalid_argument("Dynamic rendering requires a color, depth or stencil format");
    }
  }

  const std::optional<DepthBiasInfo> depth_bias = SetDepthBias();
  if (depth_bias.has_value() &&
      (!std::isfinite(depth_bias->constant_factor) || !std::isfinite(depth_bias->slope_factor))) {
    throw std::invalid_argument("Depth bias factors must be finite");
  }

  const std::vector<uint32_t> vertex_shader_code = LoadVertexShader();
  const std::vector<uint32_t> fragment_shader_code = LoadFragmentShader();
  if (vertex_shader_code.empty()) {
    throw std::invalid_argument("Graphics pipelines require a vertex shader");
  }
  if (fragment_shader_code.empty() && has_color_attachment) {
    throw std::invalid_argument(
        "Only pipelines without color attachments may omit fragment shaders");
  }

  // 1. shader stage
  const VkShaderModule vertex_shader_module = CreateShaderModule(vertex_shader_code);
  VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
  if (!fragment_shader_code.empty()) {
    fragment_shader_module = CreateShaderModule(fragment_shader_code);
  }

  VkPipelineShaderStageCreateInfo vert_info{};
  vert_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vert_info.module = vertex_shader_module;
  vert_info.pName = "main";

  VkPipelineShaderStageCreateInfo frag_info{};
  frag_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_info.module = fragment_shader_module;
  frag_info.pName = "main";

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages = {vert_info};
  if (fragment_shader_module != VK_NULL_HANDLE) {
    shader_stages.push_back(frag_info);
  }

  // 2. vertex input state
  const auto vertex_bindings = GetVertexBindingDescriptions();
  const auto vertex_attribute = GetVertexAttributeDescriptions();
  VkPipelineVertexInputStateCreateInfo vertex_input_state{};
  vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_state.vertexBindingDescriptionCount = vertex_bindings.size();
  vertex_input_state.pVertexBindingDescriptions = vertex_bindings.data();
  vertex_input_state.vertexAttributeDescriptionCount = vertex_attribute.size();
  vertex_input_state.pVertexAttributeDescriptions = vertex_attribute.data();

  // 3. input assembly state
  VkPipelineInputAssemblyStateCreateInfo input_assembly_state{};
  input_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly_state.primitiveRestartEnable = VK_FALSE;

  // 4. view port state
  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = 1.0f;
  viewport.height = 1.0f;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {1, 1};

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.pScissors = &scissor;

  // 5. rasterization state
  VkPipelineRasterizationStateCreateInfo rasterize_state{};
  rasterize_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterize_state.depthClampEnable = VK_FALSE;
  rasterize_state.rasterizerDiscardEnable = VK_FALSE;
  rasterize_state.polygonMode = VK_POLYGON_MODE_FILL;
  rasterize_state.lineWidth = 1.0f;

  rasterize_state.cullMode = SetCullMode();
  rasterize_state.frontFace = SetFrontFace();

  rasterize_state.depthBiasEnable = depth_bias.has_value() ? VK_TRUE : VK_FALSE;
  rasterize_state.depthBiasConstantFactor =
      depth_bias.has_value() ? depth_bias->constant_factor : 0.0f;
  rasterize_state.depthBiasClamp = 0.0f;
  rasterize_state.depthBiasSlopeFactor = depth_bias.has_value() ? depth_bias->slope_factor : 0.0f;

  // 6. multisample state
  VkPipelineMultisampleStateCreateInfo multisample_state{};
  multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state.sampleShadingEnable = VK_FALSE;
  multisample_state.rasterizationSamples = msaa_samples_;
  multisample_state.minSampleShading = 1.0f;
  multisample_state.pSampleMask = nullptr;
  multisample_state.alphaToCoverageEnable = VK_FALSE;
  multisample_state.alphaToOneEnable = VK_FALSE;

  // 7. dynamic state
  std::vector<VkDynamicState> dynamic_state_enables = {VK_DYNAMIC_STATE_VIEWPORT,
                                                       VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.pDynamicStates = dynamic_state_enables.data();
  dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_state_enables.size());

  // 8. color blending state
  VkPipelineColorBlendAttachmentState color_blend_attachment = SetColorBlendAttachment();

  // 9. depth testing state
  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = SetDepthTesting();
  depth_stencil.depthWriteEnable = SetDepthWriting();
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.minDepthBounds = 0.0f;  // Optional
  depth_stencil.maxDepthBounds = 1.0f;  // Optional
  depth_stencil.stencilTestEnable = VK_FALSE;
  depth_stencil.front = {};  // Optional
  depth_stencil.back = {};   // Optional

  VkPipelineColorBlendStateCreateInfo color_blending_state{};
  color_blending_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending_state.logicOpEnable = VK_FALSE;
  color_blending_state.logicOp = VK_LOGIC_OP_COPY;
  color_blending_state.attachmentCount = has_color_attachment ? 1 : 0;
  color_blending_state.pAttachments = has_color_attachment ? &color_blend_attachment : nullptr;
  color_blending_state.blendConstants[0] = 0.0f;
  color_blending_state.blendConstants[1] = 0.0f;
  color_blending_state.blendConstants[2] = 0.0f;
  color_blending_state.blendConstants[3] = 0.0f;

  // create graphic pipeline
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
  pipeline_info.pStages = shader_stages.data();
  pipeline_info.pVertexInputState = &vertex_input_state;
  pipeline_info.pInputAssemblyState = &input_assembly_state;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterize_state;
  pipeline_info.pMultisampleState = &multisample_state;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending_state;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.layout = pipeline_layout;

  // Must outlive vkCreateGraphicsPipelines() when used via pipeline_info.pNext.
  VkPipelineRenderingCreateInfo rendering_ci{};

  if (render_pass_) {
    // Traditional render-pass path
    pipeline_info.pNext = nullptr;
    pipeline_info.renderPass = render_pass_->GetRenderPass();
    pipeline_info.subpass = 0;
  } else {
    // Dynamic rendering path
    rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.pNext = nullptr;
    rendering_ci.viewMask = 0;
    rendering_ci.colorAttachmentCount =
        static_cast<uint32_t>(dynamic_rendering_info_.color_formats.size());
    rendering_ci.pColorAttachmentFormats =
        has_color_attachment ? dynamic_rendering_info_.color_formats.data() : nullptr;
    rendering_ci.depthAttachmentFormat = dynamic_rendering_info_.depth_format;
    rendering_ci.stencilAttachmentFormat = dynamic_rendering_info_.stencil_format;

    pipeline_info.pNext = &rendering_ci;
    pipeline_info.renderPass = VK_NULL_HANDLE;
    pipeline_info.subpass = 0;
  }

  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex = -1;

  VK_CHECK(vkCreateGraphicsPipelines(context_->logical_device, VK_NULL_HANDLE, 1, &pipeline_info,
                                     nullptr, &pipeline));
  vkDestroyShaderModule(context_->logical_device, fragment_shader_module, nullptr);
  vkDestroyShaderModule(context_->logical_device, vertex_shader_module, nullptr);
}

}  // namespace vulkan
}  // namespace core
