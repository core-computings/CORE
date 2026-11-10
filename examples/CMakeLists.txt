include(${CMAKE_SOURCE_DIR}/cmake/vulkan_shaders.cmake)

add_vulkan_shader_target(vulkan_example_shaders
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/vulkan
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

if(ENABLE_IO AND ENABLE_MAT)
    build_example(bitmap_demo
        SOURCES bitmap/main.cpp
        LIBRARIES io stb)
endif()

if(ENABLE_OPENCL)
    build_example(cl_program_binary_demo
        SOURCES opencl/ProgramBinary/main.cpp
        LIBRARIES opencl)
endif()

# only add glfw/draw triangle for macos
if(APPLE)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/vulkan/common)

    # 1. glfw demo
    build_example(glfw_demo
        SOURCE_GLOBS "./vulkan/GLFWWindowDemo/*.cpp"
        LIBRARIES glfw)

    # 2. draw triangle demo using vulkan
    # (glfw comes in transitively via `vulkan`, which links it PUBLIC)
    build_example(vk_triangle_demo
        SOURCE_GLOBS "./vulkan/DrawTriangleDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/DrawTriangleDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan)

    # 3. draw texture demo using vulkan
    build_example(vk_texture_demo
        SOURCE_GLOBS "./vulkan/DrawTextureDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/DrawTextureDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb)

    # 4. draw triangle demo using opengl
    # (gl_glad comes in transitively via `opengl`, which links it PUBLIC)
    build_example(gl_triangle_demo
        SOURCE_GLOBS "./opengl/GLTriangleDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLTriangleDemo
        LIBRARIES glfw opengl
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 5. draw texture demo using opengl
    # (glfw comes in transitively via `imgui`, which links it PUBLIC)
    build_example(gl_texture_demo
        SOURCE_GLOBS "./opengl/GLTextureDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLTextureDemo
        LIBRARIES opengl stb imgui
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 6. camera movement demo using opengl
    build_example(gl_camera_demo
        SOURCE_GLOBS "./opengl/GLCameraDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLCameraDemo
        LIBRARIES glfw opengl stb
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 7. draw texture demo using vulkan
    build_example(vk_depth_demo
        SOURCE_GLOBS "./vulkan/DepthTestingDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/DepthTestingDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb)

    # 8. load model demo using vulkan
    build_example(vk_model_demo
        SOURCE_GLOBS "./vulkan/LoadModelDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/LoadModelDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb tinyobjloader)

    # 9. Implement a rasterizer using compute shader + barycentric algo
    build_example(vk_rasterize_demo
        SOURCE_GLOBS "./vulkan/ComputeBarycentricRasterizer/*.cpp"
        PUBLIC_INCLUDES ./vulkan/ComputeBarycentricRasterizer
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb mat)

    # 10. depth and stencil testing demo using opengl
    build_example(gl_depth_stencil_demo
        SOURCE_GLOBS "./opengl/GLDepthStencilDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLDepthStencilDemo
        LIBRARIES glfw opengl stb
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 11. blend demo using opengl
    build_example(gl_blend_demo
        SOURCE_GLOBS "./opengl/GLBlendDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLBlendDemo
        LIBRARIES glfw opengl stb
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 12. framebuffer demo using opengl
    build_example(gl_framebuffer_demo
        SOURCE_GLOBS "./opengl/GLFrameBufferDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLFrameBufferDemo
        LIBRARIES glfw opengl stb
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 13. metal triangle demo
    # (glfw comes in transitively via `core_metal`, which links it PUBLIC)
    build_example(metal_triangle_demo
        SOURCE_GLOBS "./metal/MetalTriangleDemo/main.cpp"
        LIBRARIES core_metal)

    # 14. metal texture demo
    build_example(metal_texture_demo
        SOURCE_GLOBS "./metal/MetalTextureDemo/main.cpp"
        LIBRARIES core_metal stb)

    # 15. light demo using opengl
    # (glfw comes in transitively via `imgui`, which links it PUBLIC)
    build_example(gl_light_demo
        SOURCE_GLOBS "./opengl/GLLightDemo/*.cpp"
        PUBLIC_INCLUDES ./opengl/GLLightDemo
        LIBRARIES opengl stb imgui
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 16. cubemap demo using opengl
    build_example(gl_cubemap_demo
        SOURCE_GLOBS "./opengl/GLCubeMapDemo/*.cpp"
        LIBRARIES opengl stb imgui
        DEFINITIONS GL_SILENCE_DEPRECATION)

    # 17. mipmap demo using vulkan
    build_example(vk_mipmap_demo
        SOURCE_GLOBS "./vulkan/MipmapDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/MipmapDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb tinyobjloader imgui)

    # 18. multi-sampling demo using vulkan
    build_example(vk_multi_sampling_demo
        SOURCE_GLOBS "./vulkan/MultiSamplingDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/MultiSamplingDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb tinyobjloader imgui)

    # 19. cubemap demo using vulkan
    build_example(vk_cubemap_demo
        SOURCE_GLOBS "./vulkan/CubeMapDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/CubeMapDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb tinyobjloader imgui io)

    # 20. galaxy rotation demo using vulkan
    build_example(vk_galaxy_demo
        SOURCE_GLOBS "./vulkan/GalaxyDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/GalaxyDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan)

    # 21. physics simulation demo using vulkan
    build_example(vk_physics_demo
        SOURCE_GLOBS "./vulkan/PhysicsDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/PhysicsDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan)

    # 22. third-person car control demo using vulkan
    build_example(vk_car_control_demo
        SOURCE_GLOBS "./vulkan/CarControlDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/CarControlDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb)

    # 23. first-person shooter demo using vulkan
    build_example(vk_fps_demo
        SOURCE_GLOBS "./vulkan/FPSShooterDemo/*.cpp"
        PUBLIC_INCLUDES ./vulkan/FPSShooterDemo
        PRIVATE_INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/shaders
        DEPENDS vulkan_example_shaders
        LIBRARIES vulkan stb tinyobjloader
        DEFINITIONS FPS_ASSET_DIR="${CMAKE_CURRENT_LIST_DIR}/data/assets")

endif()
