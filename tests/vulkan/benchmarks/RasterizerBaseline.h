#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ComputeBarycentric.h"

namespace core::test {

using Rasterizer = example::ComputeBarycentric;
using Triangle = Rasterizer::Triangle;

inline bool RasterizerCovers(const Triangle& triangle, int x, int y) {
  if (x < std::min({triangle.a.x, triangle.b.x, triangle.c.x}) ||
      x > std::max({triangle.a.x, triangle.b.x, triangle.c.x}) ||
      y < std::min({triangle.a.y, triangle.b.y, triangle.c.y}) ||
      y > std::max({triangle.a.y, triangle.b.y, triangle.c.y})) {
    return false;
  }
  const glm::ivec2 vertices[] = {triangle.a, triangle.b, triangle.c,
                                 triangle.a};
  for (int edge = 0; edge < 3; ++edge) {
    const int64_t dx =
        static_cast<int64_t>(vertices[edge + 1].x) - vertices[edge].x;
    const int64_t dy =
        static_cast<int64_t>(vertices[edge + 1].y) - vertices[edge].y;
    if (dx * (static_cast<int64_t>(y) - vertices[edge].y) -
            dy * (static_cast<int64_t>(x) - vertices[edge].x) <
        0) {
      return false;
    }
  }
  return true;
}

class RasterizerBaseline : public Rasterizer {
 public:
  RasterizerBaseline(vulkan::VulkanContext* context, int width, int height)
      : Rasterizer(context, width, height), width_(width), height_(height) {}

  void Run(VkCommandBuffer command_buffer, const Triangle& triangle) {
    for (const glm::ivec2& vertex : {triangle.a, triangle.b, triangle.c}) {
      if (vertex.x < -16383 || vertex.x > 16383 || vertex.y < -16383 ||
          vertex.y > 16383) {
        throw std::invalid_argument(
            "Rasterizer coordinates must be between -16383 and 16383");
      }
    }
    const PushConstants parameters{.width = width_,
                                   .height = height_,
                                   .vertex_a = triangle.a,
                                   .vertex_b = triangle.b,
                                   .vertex_c = triangle.c};
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set_, 0,
                            nullptr);
    vkCmdPushConstants(command_buffer, pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(sizeof(parameters)), &parameters);

    const uint32_t group_x = (static_cast<uint32_t>(width_) - 1) / 16U + 1;
    const uint32_t group_y = (static_cast<uint32_t>(height_) - 1) / 16U + 1;
    vkCmdDispatch(command_buffer, group_x, group_y, 1);
  }

 private:
  struct PushConstants {
    int width;
    int height;
    glm::ivec2 vertex_a;
    glm::ivec2 vertex_b;
    glm::ivec2 vertex_c;
  };
  static_assert(sizeof(PushConstants) == 32);

  std::vector<VkPushConstantRange> GetPushConstantRanges() const override {
    return {{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .offset = 0,
             .size = static_cast<uint32_t>(sizeof(PushConstants))}};
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> code =
#include "RasterizerBaseline.comp.spv"
        ;
    return code;
  }

  int width_;
  int height_;
};

}
