#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace core::test {

inline std::vector<float> GaussianInput(int width, int height) {
  std::vector<float> input(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>((index * 73 + index / 17) % 4093) -
                                      2046) / 16.0f;
  }
  return input;
}

inline std::vector<float> GaussianReference(const std::vector<float>& input, int width,
                                           int height) {
  constexpr std::array<float, 9> kWeights = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f,
                                            0.125f, 0.0625f, 0.125f, 0.0625f};
  std::vector<float> output = input;
  for (int row = 1; row < height - 1; ++row) {
    for (int col = 1; col < width - 1; ++col) {
      float sum = 0.0f;
      size_t weight = 0;
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          sum += input[static_cast<size_t>((row + y) * width + col + x)] * kWeights[weight];
          ++weight;
        }
      }
      output[static_cast<size_t>(row * width + col)] = sum;
    }
  }
  return output;
}

}
