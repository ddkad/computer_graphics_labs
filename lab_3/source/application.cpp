#include "application.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include <imgui.h>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "math3d.hpp"
#include "octahedron.hpp"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

#ifndef TEXTURE_DIR
#define TEXTURE_DIR "textures/"
#endif

namespace application {

namespace {

using math::Mat4;
using math::Vec3;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

struct Texture {
	std::string name;
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	uint32_t mip_levels = 1;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

struct SceneData {
	float view_proj[16];
	float camera_position[4];
	float ambient[4];
	float dir_direction[4];
	float dir_color[4];
	uint32_t counts[4];
};
static_assert(sizeof(SceneData) == 144, "Unexpected scene uniform layout");

struct ModelData {
	float model[16];
	float normal_matrix[16];
	float albedo[4];
	float specular[4];
};
static_assert(sizeof(ModelData) == 160, "Unexpected model uniform layout");

constexpr uint32_t FlagNonTrivialSampling = 1u << 0;
constexpr uint32_t FlagBoxBlur = 1u << 1;

struct PushConstants {
	uint32_t flags;
	float time;
	float amplitude;
	int32_t blur_radius;
	float tint[4];
};
static_assert(sizeof(PushConstants) == 32, "Unexpected push constants layout");

struct GpuPointLight {
	float position[4];
	float color[4];
};
static_assert(sizeof(GpuPointLight) == 32, "Unexpected point light layout");

struct GpuSpotLight {
	float position[4];
	float direction[4];
	float color[4];
	float cone[4];
};
static_assert(sizeof(GpuSpotLight) == 64, "Unexpected spot light layout");

struct Mesh {
	uint32_t first_index = 0;
	uint32_t index_count = 0;
	int32_t vertex_offset = 0;
};

enum MeshId : int { MeshOctahedron = 0, MeshPlane = 1, MeshCount = 2 };

struct Material {
	float albedo[3];
	float specular[3];
	float shininess;
};

struct Sampling {
	bool nontrivial = false;
	bool box_blur = false;
	int blur_radius = 2;
	float amplitude = 1.0f;
	float tint[3] = { 1.0f, 0.85f, 0.7f };
};

struct Object {
	const char* name;
	int mesh;
	float position[3];
	float rotation_deg[3];
	float scale[3];
	Material material;
	const char* default_texture;
	Sampling sampling;
	int texture = 0;
};

enum Projection : int { Perspective = 0, Orthographic = 1 };
enum Trajectory : int { Lissajous = 0, Trefoil = 1, Spiral = 2 };

struct Camera {
	int projection = Perspective;
	float position[3] = { 0.0f, 2.5f, 11.0f };
	float yaw_deg = 0.0f;
	float pitch_deg = -8.0f;
	float roll_deg = 0.0f;
	float fov_deg = 60.0f;
	float ortho_size = 6.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float move_speed = 6.0f;
	float mouse_sensitivity = 0.2f;
};

struct DirectionalLight {
	bool enabled = true;
	float yaw_deg = -40.0f;
	float pitch_deg = -45.0f;
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 0.5f;
};

struct PointLight {
	float position[3] = { 0.0f, 2.0f, 3.0f };
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 40.0f;
};

struct SpotLight {
	float position[3] = { 0.0f, 6.0f, 2.0f };
	float target[3] = { 0.0f, -2.5f, 0.0f };
	float color[3] = { 1.0f, 1.0f, 0.9f };
	float intensity = 80.0f;
	float inner_deg = 12.0f;
	float outer_deg = 25.0f;
};

struct Animation {
	bool playing = true;
	float speed = 1.0f;
	int trajectory = Lissajous;
	float radius = 2.5f;
	float height = 0.7f;
	int freq[3] = { 3, 2, 1 };
	float phase_deg = 90.0f;
	float spin_speed_deg = 60.0f;
	float spin_axis[3] = { 0.4f, 1.0f, 0.2f };
	float clock = 0.0f;
	float spin_angle = 0.0f;
};

struct State {
	Camera camera;
	DirectionalLight directional;
	float ambient_color[3] = { 0.6f, 0.65f, 1.0f };
	float ambient_intensity = 0.12f;
	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
	Animation animation;
	float effect_speed = 1.0f;
	float effect_clock = 0.0f;
	int selected_object = 0;
	double last_time = -1.0;
};

constexpr size_t max_textures = 32;
constexpr float plane_uv_tiling = 8.0f;
constexpr uint32_t initial_light_capacity = 8;

State state;
std::vector<Object> objects;
std::vector<Texture> textures;
Mesh meshes[MeshCount];

Buffer vertex_buffer;
Buffer index_buffer;
Buffer scene_buffer;
Buffer point_buffer;
Buffer spot_buffer;
Buffer model_buffer;
VkDeviceSize model_stride = 0;

VkCommandPool transfer_command_pool = VK_NULL_HANDLE;
VkSampler texture_sampler = VK_NULL_HANDLE;

VkShaderModule shader_vert = VK_NULL_HANDLE;
VkShaderModule shader_frag = VK_NULL_HANDLE;
VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

Vec3 toVec3(const float v[3]) {
	return { v[0], v[1], v[2] };
}

Vec3 directionFromAngles(float yaw_deg, float pitch_deg) {
	const float yaw = math::radians(yaw_deg);
	const float pitch = math::radians(pitch_deg);
	return {
		-std::sin(yaw) * std::cos(pitch),
		std::sin(pitch),
		-std::cos(yaw) * std::cos(pitch),
	};
}

Vec3 trajectoryOffset(const Animation& a, float t) {
	switch (a.trajectory) {
	case Lissajous: {
		const float phase = math::radians(a.phase_deg);
		return {
			a.radius * std::sin(float(a.freq[0]) * t + phase),
			a.height * std::sin(float(a.freq[1]) * t),
			a.radius * std::sin(float(a.freq[2]) * t),
		};
	}
	case Trefoil: {
		const float k = a.radius / 3.0f;
		return {
			k * (std::sin(t) + 2.0f * std::sin(2.0f * t)),
			-a.height * std::sin(3.0f * t),
			k * (std::cos(t) - 2.0f * std::cos(2.0f * t)),
		};
	}
	case Spiral:
	default: {
		const float r = a.radius * (0.65f + 0.35f * std::sin(0.3f * t));
		return {
			r * std::cos(t),
			a.height * std::sin(float(a.freq[1]) * 0.5f * t),
			r * std::sin(t),
		};
	}
	}
}

Mat4 modelMatrix(int index) {
	const Object& o = objects[size_t(index)];

	Vec3 offset = {};
	Mat4 spin = math::identity();

	if (index == 0) {
		offset = trajectoryOffset(state.animation, state.animation.clock);
		spin = math::rotateAxis(toVec3(state.animation.spin_axis), state.animation.spin_angle);
	}

	const Mat4 T = math::translate(toVec3(o.position) + offset);
	const Mat4 R = math::rotateZ(math::radians(o.rotation_deg[2])) *
	               math::rotateY(math::radians(o.rotation_deg[1])) *
	               math::rotateX(math::radians(o.rotation_deg[0]));
	const Mat4 S = math::scale(toVec3(o.scale));

	return T * R * spin * S;
}

Mat4 buildView(const Camera& c) {
	return math::rotateZ(-math::radians(c.roll_deg)) *
	       math::rotateX(-math::radians(c.pitch_deg)) *
	       math::rotateY(-math::radians(c.yaw_deg)) *
	       math::translate(toVec3(c.position) * -1.0f);
}

Mat4 buildProjection(const Camera& c, float aspect) {
	if (c.projection == Perspective) {
		return math::perspective(math::radians(c.fov_deg), aspect, c.near_plane, c.far_plane);
	}
	return math::orthographic(c.ortho_size, aspect, c.near_plane, c.far_plane);
}

float currentAspect() {
	const auto& extent = graphics::internal::context.swapchain_extent;
	return extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;
}

void showMatrix(const char* label, const Mat4& m) {
	ImGui::TextUnformatted(label);
	for (int row = 0; row < 4; ++row) {
		ImGui::Text("%8.3f %8.3f %8.3f %8.3f", m.m[0][row], m.m[1][row], m.m[2][row], m.m[3][row]);
	}
}

void setupInitialState() {
	State& s = state;

	s.point_lights.clear();
	s.spot_lights.clear();

	PointLight warm;
	warm.position[0] = -4.5f; warm.position[1] = 1.5f; warm.position[2] = 3.0f;
	warm.color[0] = 1.0f; warm.color[1] = 0.65f; warm.color[2] = 0.35f;
	warm.intensity = 40.0f;
	s.point_lights.push_back(warm);

	PointLight cold;
	cold.position[0] = 4.5f; cold.position[1] = 0.5f; cold.position[2] = 3.0f;
	cold.color[0] = 0.35f; cold.color[1] = 0.55f; cold.color[2] = 1.0f;
	cold.intensity = 40.0f;
	s.point_lights.push_back(cold);

	s.spot_lights.push_back(SpotLight{});

	Sampling animated_sampling;
	animated_sampling.nontrivial = true;

	objects.clear();
	objects.push_back({ "Octahedron (animated)", MeshOctahedron,
	                    { 0.0f, 0.0f, 0.0f }, { 20.0f, 30.0f, 0.0f }, { 1.2f, 1.2f, 1.2f },
	                    { { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 64.0f },
	                    "uv_test", animated_sampling });
	objects.push_back({ "Floor", MeshPlane,
	                    { 0.0f, -2.5f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 10.0f, 1.0f, 10.0f },
	                    { { 1.0f, 1.0f, 1.0f }, { 0.25f, 0.25f, 0.25f }, 16.0f },
	                    "checker", Sampling{} });
	objects.push_back({ "Back wall", MeshPlane,
	                    { 0.0f, 7.5f, -10.0f }, { 90.0f, 0.0f, 0.0f }, { 10.0f, 1.0f, 10.0f },
	                    { { 1.0f, 1.0f, 1.0f }, { 0.1f, 0.1f, 0.1f }, 8.0f },
	                    "bricks", Sampling{} });
	objects.push_back({ "Octahedron A", MeshOctahedron,
	                    { -4.0f, -1.7f, -1.0f }, { 0.0f, 20.0f, 0.0f }, { 0.8f, 0.8f, 0.8f },
	                    { { 1.0f, 1.0f, 1.0f }, { 0.6f, 0.6f, 0.6f }, 32.0f },
	                    "wood", Sampling{} });
	objects.push_back({ "Octahedron B", MeshOctahedron,
	                    { 4.0f, -1.2f, -2.0f }, { 0.0f, -35.0f, 0.0f }, { 1.3f, 1.3f, 1.3f },
	                    { { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 128.0f },
	                    "marble", Sampling{} });
}

int findTexture(const char* name) {
	for (size_t i = 0; i < textures.size(); ++i) {
		if (textures[i].name == name) {
			return int(i);
		}
	}
	return 0;
}

bool createShaderModule(const std::string& path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file: " << path
		          << "\n(compile shaders with glslc first, see CMakeLists.txt)\n";
		return false;
	}

	const size_t size = size_t(file.tellg());
	if (size == 0 || size % 4 != 0) {
		std::cerr << "Invalid SPIR-V file: " << path << '\n';
		return false;
	}

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return false;
	}

	return true;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result{};
	auto& allocator = graphics::internal::context.allocator;

	Buffer created;
	if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info,
	                    &created.buffer, &created.allocation, &result) != VK_SUCCESS ||
	    result.pMappedData == nullptr) {
		std::cerr << "Failed to create Vulkan buffer\n";
		if (created.buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(allocator, created.buffer, created.allocation);
		}
		return false;
	}

	created.mapped = result.pMappedData;
	created.size = size;

	if (data != nullptr) {
		std::memcpy(created.mapped, data, size_t(size));
	} else {
		std::memset(created.mapped, 0, size_t(size));
	}
	vmaFlushAllocation(allocator, created.allocation, 0, VK_WHOLE_SIZE);

	out = created;
	return true;
}

bool createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	Buffer created;
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &allocation_info,
	                    &created.buffer, &created.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device-local Vulkan buffer\n";
		return false;
	}

	created.size = size;
	out = created;
	return true;
}

void writeBuffer(Buffer& buffer, const void* data, VkDeviceSize size) {
	if (buffer.mapped == nullptr || size == 0 || size > buffer.size) {
		return;
	}
	std::memcpy(buffer.mapped, data, size_t(size));
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

bool submitOneTime(const std::function<void(VkCommandBuffer)>& record) {
	auto& context = graphics::internal::context;

	const VkCommandBufferAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = transfer_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	VkCommandBuffer cb = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(context.device, &alloc_info, &cb) != VK_SUCCESS) {
		std::cerr << "Failed to allocate transfer command buffer\n";
		return false;
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cb, &begin);
	record(cb);
	vkEndCommandBuffer(cb);

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
	};

	const VkResult result = vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
	if (result == VK_SUCCESS) {
		vkQueueWaitIdle(context.graphics_queue);
	} else {
		std::cerr << "Failed to submit transfer commands\n";
	}

	vkFreeCommandBuffers(context.device, transfer_command_pool, 1, &cb);
	return result == VK_SUCCESS;
}

bool createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data, staging)) {
		return false;
	}

	Buffer device;
	if (!createDeviceBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, device)) {
		destroyBuffer(staging);
		return false;
	}

	const bool ok = submitOneTime([&](VkCommandBuffer cb) {
		const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
		vkCmdCopyBuffer(cb, staging.buffer, device.buffer, 1, &region);
	});

	destroyBuffer(staging);

	if (!ok) {
		destroyBuffer(device);
		return false;
	}

	out = device;
	return true;
}

void imageBarrier(VkCommandBuffer cb, VkImage image,
                  VkImageLayout old_layout, VkImageLayout new_layout,
                  VkAccessFlags src_access, VkAccessFlags dst_access,
                  VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                  uint32_t base_mip, uint32_t mip_count) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip,
			.levelCount = mip_count,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool readFile(const std::filesystem::path& path, std::vector<unsigned char>& data) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		return false;
	}

	const std::streamsize size = file.tellg();
	if (size <= 0) {
		return false;
	}

	data.resize(size_t(size));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(data.data()), size);
	return bool(file);
}

bool createTexture(const std::filesystem::path& path, Texture& out) {
	auto& context = graphics::internal::context;

	std::vector<unsigned char> file_data;
	if (!readFile(path, file_data)) {
		std::cerr << "Failed to read image file: " << path.string() << '\n';
		return false;
	}

	int width = 0, height = 0, channels = 0;
	stbi_uc* pixels = stbi_load_from_memory(file_data.data(), int(file_data.size()),
	                                        &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr || width <= 0 || height <= 0) {
		std::cerr << "Failed to decode image " << path.string() << ": " << stbi_failure_reason() << '\n';
		if (pixels != nullptr) {
			stbi_image_free(pixels);
		}
		return false;
	}

	const VkDeviceSize image_size = VkDeviceSize(width) * VkDeviceSize(height) * 4;

	Buffer staging;
	const bool staged = createBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, pixels, staging);
	stbi_image_free(pixels);
	if (!staged) {
		return false;
	}

	const VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

	VkFormatProperties format_properties{};
	vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &format_properties);

	const VkFormatFeatureFlags mip_features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
	                                          VK_FORMAT_FEATURE_BLIT_SRC_BIT |
	                                          VK_FORMAT_FEATURE_BLIT_DST_BIT;

	uint32_t mip_levels = 1;
	if ((format_properties.optimalTilingFeatures & mip_features) == mip_features) {
		mip_levels = uint32_t(std::floor(std::log2(float(std::max(width, height))))) + 1;
	}

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { uint32_t(width), uint32_t(height), 1 },
		.mipLevels = mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		         VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	Texture texture;
	texture.name = path.stem().string();
	texture.mip_levels = mip_levels;

	if (vmaCreateImage(context.allocator, &image_info, &allocation_info,
	                   &texture.image, &texture.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image for " << path.string() << '\n';
		destroyBuffer(staging);
		return false;
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = mip_levels,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view_info, nullptr, &texture.view) != VK_SUCCESS) {
		std::cerr << "Failed to create image view for " << path.string() << '\n';
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
		destroyBuffer(staging);
		return false;
	}

	const bool uploaded = submitOneTime([&](VkCommandBuffer cb) {
		imageBarrier(cb, texture.image,
		             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		             0, VK_ACCESS_TRANSFER_WRITE_BIT,
		             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		             0, mip_levels);

		const VkBufferImageCopy region = {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.imageOffset = { 0, 0, 0 },
			.imageExtent = { uint32_t(width), uint32_t(height), 1 },
		};
		vkCmdCopyBufferToImage(cb, staging.buffer, texture.image,
		                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		int32_t mip_width = width;
		int32_t mip_height = height;

		for (uint32_t i = 1; i < mip_levels; ++i) {
			imageBarrier(cb, texture.image,
			             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
			             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			             i - 1, 1);

			const int32_t next_width = std::max(mip_width / 2, 1);
			const int32_t next_height = std::max(mip_height / 2, 1);

			const VkImageBlit blit = {
				.srcSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = i - 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffsets = { { 0, 0, 0 }, { mip_width, mip_height, 1 } },
				.dstSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = i,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.dstOffsets = { { 0, 0, 0 }, { next_width, next_height, 1 } },
			};
			vkCmdBlitImage(cb, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			               texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			               1, &blit, VK_FILTER_LINEAR);

			imageBarrier(cb, texture.image,
			             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			             VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
			             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			             i - 1, 1);

			mip_width = next_width;
			mip_height = next_height;
		}

		imageBarrier(cb, texture.image,
		             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		             mip_levels - 1, 1);
	});

	destroyBuffer(staging);

	if (!uploaded) {
		vkDestroyImageView(context.device, texture.view, nullptr);
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
		return false;
	}

	out = texture;
	return true;
}

bool loadTextures() {
	namespace fs = std::filesystem;

	const char* extensions[] = { ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".pic", ".ppm", ".pgm" };

	std::error_code error;
	fs::directory_iterator iterator(TEXTURE_DIR, error);
	if (error) {
		std::cerr << "Failed to open texture directory: " << TEXTURE_DIR << '\n';
		return false;
	}

	std::vector<fs::path> files;
	for (const fs::directory_entry& entry : iterator) {
		if (!entry.is_regular_file()) {
			continue;
		}

		std::string extension = entry.path().extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(),
		               [](unsigned char c) { return char(std::tolower(c)); });

		for (const char* allowed : extensions) {
			if (extension == allowed) {
				files.push_back(entry.path());
				break;
			}
		}
	}

	std::sort(files.begin(), files.end());

	for (const fs::path& file : files) {
		if (textures.size() >= max_textures) {
			break;
		}

		Texture texture;
		if (createTexture(file, texture)) {
			textures.push_back(texture);
		}
	}

	if (textures.empty()) {
		std::cerr << "No textures were loaded from " << TEXTURE_DIR << '\n';
		return false;
	}

	return true;
}

bool createSampler() {
	const VkSamplerCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_FALSE,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
	};

	if (vkCreateSampler(graphics::internal::context.device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create texture sampler\n";
		return false;
	}

	return true;
}

void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void writeTextureSet(VkDescriptorSet set, const Texture& texture) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = model_buffer.buffer,
		.offset = 0,
		.range = sizeof(ModelData),
	};

	const VkDescriptorImageInfo image_info = {
		.sampler = texture_sampler,
		.imageView = texture.view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};

	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &buffer_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = set,
			.dstBinding = 1,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_info,
		},
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 2, writes, 0, nullptr);
}

bool growStorageBuffer(Buffer& buffer, uint32_t binding, VkDeviceSize required) {
	if (buffer.size >= required) {
		return true;
	}

	Buffer grown;
	const VkDeviceSize new_size = std::max<VkDeviceSize>(buffer.size * 2, required);
	if (!createBuffer(new_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, grown)) {
		return false;
	}

	destroyBuffer(buffer);
	buffer = grown;
	writeDescriptor(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(model_bindings) / sizeof(model_bindings[0]),
		.pBindings = model_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	const uint32_t texture_count = uint32_t(textures.size());

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = texture_count },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = texture_count },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + texture_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_alloc = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_alloc, &scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	writeDescriptor(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
	writeDescriptor(scene_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer);
	writeDescriptor(scene_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer);

	for (Texture& texture : textures) {
		const VkDescriptorSetAllocateInfo model_alloc = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &model_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &model_alloc, &texture.set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate material descriptor set\n";
			return false;
		}

		writeTextureSet(texture.set, texture);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	if (!createShaderModule(std::string(SHADER_DIR) + "octahedron.vert.spv", shader_vert) ||
	    !createShaderModule(std::string(SHADER_DIR) + "octahedron.frag.spv", shader_frag)) {
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };

	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shader_vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = shader_frag,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, position)) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, normal)) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, uv)) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
	                              nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

void destroyResources() {
	auto& context = graphics::internal::context;

	if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, pipeline, nullptr);
	if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (scene_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
	if (model_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	if (shader_vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, shader_vert, nullptr);
	if (shader_frag != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, shader_frag, nullptr);
	if (texture_sampler != VK_NULL_HANDLE) vkDestroySampler(context.device, texture_sampler, nullptr);

	for (Texture& texture : textures) {
		if (texture.view != VK_NULL_HANDLE) vkDestroyImageView(context.device, texture.view, nullptr);
		if (texture.image != VK_NULL_HANDLE) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	}
	textures.clear();

	destroyBuffer(model_buffer);
	destroyBuffer(scene_buffer);
	destroyBuffer(point_buffer);
	destroyBuffer(spot_buffer);
	destroyBuffer(vertex_buffer);
	destroyBuffer(index_buffer);

	if (transfer_command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(context.device, transfer_command_pool, nullptr);

	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	scene_set = VK_NULL_HANDLE;
	scene_set_layout = VK_NULL_HANDLE;
	model_set_layout = VK_NULL_HANDLE;
	texture_sampler = VK_NULL_HANDLE;
	transfer_command_pool = VK_NULL_HANDLE;
	shader_vert = shader_frag = VK_NULL_HANDLE;
}

void updateCamera(Camera& c, float dt) {
	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		c.yaw_deg -= io.MouseDelta.x * c.mouse_sensitivity;
		c.pitch_deg = std::clamp(c.pitch_deg - io.MouseDelta.y * c.mouse_sensitivity, -89.0f, 89.0f);

		if (c.yaw_deg > 180.0f) c.yaw_deg -= 360.0f;
		if (c.yaw_deg < -180.0f) c.yaw_deg += 360.0f;
	}

	if (io.WantTextInput) {
		return;
	}

	const Vec3 forward = directionFromAngles(c.yaw_deg, c.pitch_deg);
	const float yaw = math::radians(c.yaw_deg);
	const Vec3 right = { std::cos(yaw), 0.0f, -std::sin(yaw) };
	const Vec3 up = { 0.0f, 1.0f, 0.0f };

	Vec3 move = {};
	if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + forward;
	if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - forward;
	if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + right;
	if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - right;
	if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + up;
	if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - up;

	if (math::length(move) > 0.0f) {
		const float boost = ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0f : 1.0f;
		const Vec3 delta = math::normalize(move) * (c.move_speed * boost * dt);
		c.position[0] += delta.x;
		c.position[1] += delta.y;
		c.position[2] += delta.z;
	}
}

void drawCameraUi(Camera& c) {
	ImGui::RadioButton("Perspective", &c.projection, Perspective);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &c.projection, Orthographic);

	ImGui::DragFloat3("Position", c.position, 0.05f);
	ImGui::SliderFloat("Yaw (deg)", &c.yaw_deg, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch (deg)", &c.pitch_deg, -89.0f, 89.0f);
	ImGui::SliderFloat("Roll (deg)", &c.roll_deg, -180.0f, 180.0f);

	if (c.projection == Perspective) {
		ImGui::SliderFloat("FOV (deg)", &c.fov_deg, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Ortho size (half height)", &c.ortho_size, 1.0f, 20.0f);
	}

	ImGui::SliderFloat("Near plane", &c.near_plane, 0.01f, 5.0f);
	ImGui::SliderFloat("Far plane", &c.far_plane, 10.0f, 300.0f);
	ImGui::SliderFloat("Move speed", &c.move_speed, 0.5f, 30.0f);
	ImGui::SliderFloat("Mouse sensitivity", &c.mouse_sensitivity, 0.05f, 1.0f);

	if (ImGui::Button("Reset camera")) {
		c = Camera{};
	}

	ImGui::TextDisabled("Hold RMB + move mouse: look around");
	ImGui::TextDisabled("W/A/S/D: move, Q/E: down/up, Shift: faster");
}

void drawLightingUi(State& s) {
	ImGui::ColorEdit3("Ambient color", s.ambient_color);
	ImGui::SliderFloat("Ambient intensity", &s.ambient_intensity, 0.0f, 1.0f);

	ImGui::Separator();
	ImGui::TextUnformatted("Directional light");
	DirectionalLight& d = s.directional;
	ImGui::Checkbox("Enabled##directional", &d.enabled);
	ImGui::SliderFloat("Yaw (deg)##directional", &d.yaw_deg, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch (deg)##directional", &d.pitch_deg, -89.0f, 89.0f);
	ImGui::ColorEdit3("Color##directional", d.color);
	ImGui::SliderFloat("Intensity##directional", &d.intensity, 0.0f, 5.0f);
}

void drawPointLightsUi(State& s) {
	if (ImGui::Button("Add point light")) {
		s.point_lights.push_back(PointLight{});
	}

	int remove_index = -1;
	for (int i = 0; i < int(s.point_lights.size()); ++i) {
		PointLight& l = s.point_lights[size_t(i)];

		char label[48];
		std::snprintf(label, sizeof(label), "Point light %d", i + 1);

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", l.position, 0.05f);
			ImGui::ColorEdit3("Color", l.color);
			ImGui::DragFloat("Intensity", &l.intensity, 0.5f, 0.0f, 100000.0f);

			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		s.point_lights.erase(s.point_lights.begin() + remove_index);
	}
}

void drawSpotLightsUi(State& s) {
	if (ImGui::Button("Add spot light")) {
		s.spot_lights.push_back(SpotLight{});
	}

	int remove_index = -1;
	for (int i = 0; i < int(s.spot_lights.size()); ++i) {
		SpotLight& l = s.spot_lights[size_t(i)];

		char label[48];
		std::snprintf(label, sizeof(label), "Spot light %d", i + 1);

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", l.position, 0.05f);
			ImGui::DragFloat3("Target", l.target, 0.05f);
			ImGui::ColorEdit3("Color", l.color);
			ImGui::DragFloat("Intensity", &l.intensity, 0.5f, 0.0f, 100000.0f);

			const bool inner_changed = ImGui::SliderFloat("Inner angle (deg)", &l.inner_deg, 0.0f, 89.0f);
			const bool outer_changed = ImGui::SliderFloat("Outer angle (deg)", &l.outer_deg, 0.0f, 90.0f);
			if (inner_changed && l.inner_deg > l.outer_deg) {
				l.outer_deg = l.inner_deg;
			}
			if (outer_changed && l.outer_deg < l.inner_deg) {
				l.inner_deg = l.outer_deg;
			}

			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		s.spot_lights.erase(s.spot_lights.begin() + remove_index);
	}
}

void drawObjectsUi(State& s) {
	if (ImGui::BeginCombo("Object", objects[size_t(s.selected_object)].name)) {
		for (int i = 0; i < int(objects.size()); ++i) {
			if (ImGui::Selectable(objects[size_t(i)].name, i == s.selected_object)) {
				s.selected_object = i;
			}
		}
		ImGui::EndCombo();
	}

	Object& o = objects[size_t(s.selected_object)];

	ImGui::DragFloat3("Position", o.position, 0.05f);
	ImGui::SliderFloat3("Rotation (deg)", o.rotation_deg, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", o.scale, 0.1f, 15.0f);

	ImGui::Separator();
	ImGui::TextUnformatted("Texture");
	if (ImGui::BeginCombo("Albedo texture", textures[size_t(o.texture)].name.c_str())) {
		for (int i = 0; i < int(textures.size()); ++i) {
			ImGui::PushID(i);
			if (ImGui::Selectable(textures[size_t(i)].name.c_str(), i == o.texture)) {
				o.texture = i;
			}
			ImGui::PopID();
		}
		ImGui::EndCombo();
	}
	ImGui::TextDisabled("Loaded textures: %d (folder textures/)", int(textures.size()));

	ImGui::Separator();
	ImGui::TextUnformatted("Material (Blinn-Phong)");
	ImGui::ColorEdit3("Albedo tint", o.material.albedo);
	ImGui::ColorEdit3("Specular color", o.material.specular);
	ImGui::SliderFloat("Shininess", &o.material.shininess, 1.0f, 512.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

	ImGui::Separator();
	ImGui::TextUnformatted("Texture sampling (push constants)");
	Sampling& sm = o.sampling;
	ImGui::Checkbox("Non-trivial sampling", &sm.nontrivial);
	if (sm.nontrivial) {
		ImGui::SliderFloat("Shift amplitude", &sm.amplitude, 0.0f, 3.0f);
		ImGui::ColorEdit3("Layer modulation", sm.tint);
	}
	ImGui::Checkbox("Box blur (convolution)", &sm.box_blur);
	if (sm.box_blur) {
		ImGui::SliderInt("Blur radius (texels)", &sm.blur_radius, 1, 8);
		ImGui::TextDisabled("Kernel: %dx%d, %d samples", 2 * sm.blur_radius + 1, 2 * sm.blur_radius + 1,
		                    (2 * sm.blur_radius + 1) * (2 * sm.blur_radius + 1));
	}
	ImGui::SliderFloat("Effect speed", &s.effect_speed, 0.0f, 5.0f);
}

void drawAnimationUi(Animation& a) {
	if (ImGui::Button(a.playing ? "Pause" : "Play")) {
		a.playing = !a.playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		a.clock = 0.0f;
		a.spin_angle = 0.0f;
	}

	ImGui::SliderFloat("Speed", &a.speed, 0.0f, 5.0f);

	const char* trajectories[] = { "Lissajous curve", "Trefoil knot", "Breathing spiral" };
	ImGui::Combo("Trajectory", &a.trajectory, trajectories, 3);

	ImGui::SliderFloat("Radius", &a.radius, 0.5f, 8.0f);
	ImGui::SliderFloat("Height amplitude", &a.height, 0.0f, 4.0f);

	if (a.trajectory == Lissajous) {
		ImGui::SliderInt3("Frequencies XYZ", a.freq, 1, 7);
		ImGui::SliderFloat("Phase (deg)", &a.phase_deg, 0.0f, 360.0f);
	} else if (a.trajectory == Spiral) {
		ImGui::SliderInt("Vertical frequency", &a.freq[1], 1, 7);
	}

	ImGui::SliderFloat("Spin speed (deg/s)", &a.spin_speed_deg, -360.0f, 360.0f);
	ImGui::DragFloat3("Spin axis", a.spin_axis, 0.01f, -1.0f, 1.0f);
}

}

bool initialize() {
	setupInitialState();

	auto& context = graphics::internal::context;

	const VkCommandPoolCreateInfo command_pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool_info, nullptr, &transfer_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create transfer command pool\n";
		destroyResources();
		return false;
	}

	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;

	std::vector<Vertex> mesh_vertices;
	std::vector<uint16_t> mesh_indices;

	makeOctahedron(mesh_vertices, mesh_indices);
	meshes[MeshOctahedron] = { uint32_t(indices.size()), uint32_t(mesh_indices.size()), int32_t(vertices.size()) };
	vertices.insert(vertices.end(), mesh_vertices.begin(), mesh_vertices.end());
	indices.insert(indices.end(), mesh_indices.begin(), mesh_indices.end());

	makePlane(mesh_vertices, mesh_indices, plane_uv_tiling);
	meshes[MeshPlane] = { uint32_t(indices.size()), uint32_t(mesh_indices.size()), int32_t(vertices.size()) };
	vertices.insert(vertices.end(), mesh_vertices.begin(), mesh_vertices.end());
	indices.insert(indices.end(), mesh_indices.begin(), mesh_indices.end());

	VkPhysicalDeviceProperties device_properties{};
	vkGetPhysicalDeviceProperties(context.physical_device, &device_properties);
	const VkDeviceSize alignment = std::max<VkDeviceSize>(device_properties.limits.minUniformBufferOffsetAlignment, 1);
	model_stride = (sizeof(ModelData) + alignment - 1) / alignment * alignment;

	const bool ok =
		createDeviceLocalBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
		                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) &&
		createDeviceLocalBuffer(indices.data(), indices.size() * sizeof(uint16_t),
		                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) &&
		loadTextures() &&
		createSampler() &&
		createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr, scene_buffer) &&
		createBuffer(initial_light_capacity * sizeof(GpuPointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		             nullptr, point_buffer) &&
		createBuffer(initial_light_capacity * sizeof(GpuSpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		             nullptr, spot_buffer) &&
		createBuffer(model_stride * objects.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr, model_buffer);

	if (!ok) {
		destroyResources();
		return false;
	}

	for (Object& o : objects) {
		o.texture = findTexture(o.default_texture);
	}

	if (!createDescriptors() || !createPipeline()) {
		destroyResources();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
	destroyResources();
}

void update(double time) {
	State& s = state;

	const float dt = s.last_time < 0.0 ? 0.0f : float(std::fmin(time - s.last_time, 0.1));
	s.last_time = time;

	updateCamera(s.camera, dt);

	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(470.0f, 780.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Textures - Lab 3");

	if (ImGui::CollapsingHeader("Camera")) {
		drawCameraUi(s.camera);
	}

	if (ImGui::CollapsingHeader("Ambient & directional light")) {
		drawLightingUi(s);
	}

	char header[64];

	std::snprintf(header, sizeof(header), "Point lights (%d)###point_lights", int(s.point_lights.size()));
	if (ImGui::CollapsingHeader(header)) {
		drawPointLightsUi(s);
	}

	std::snprintf(header, sizeof(header), "Spot lights (%d)###spot_lights", int(s.spot_lights.size()));
	if (ImGui::CollapsingHeader(header)) {
		drawSpotLightsUi(s);
	}

	if (ImGui::CollapsingHeader("Objects, textures & materials", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawObjectsUi(s);
	}

	if (ImGui::CollapsingHeader("Animation (main octahedron)")) {
		drawAnimationUi(s.animation);
	}

	if (s.animation.playing) {
		s.animation.clock += dt * s.animation.speed;
		s.animation.spin_angle += math::radians(s.animation.spin_speed_deg) * dt * s.animation.speed;
	}

	s.effect_clock += dt * s.effect_speed;

	if (ImGui::CollapsingHeader("Matrices")) {
		showMatrix("Model (selected object):", modelMatrix(s.selected_object));
		showMatrix("View:", buildView(s.camera));
		showMatrix("Projection:", buildProjection(s.camera, currentAspect()));
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	const State& s = state;
	VkCommandBuffer cb = fd.command_buffer;

	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);

	std::vector<GpuPointLight> point_data;
	point_data.reserve(s.point_lights.size());
	for (const PointLight& l : s.point_lights) {
		point_data.push_back({
			{ l.position[0], l.position[1], l.position[2], 1.0f },
			{ l.color[0], l.color[1], l.color[2], l.intensity },
		});
	}

	std::vector<GpuSpotLight> spot_data;
	spot_data.reserve(s.spot_lights.size());
	for (const SpotLight& l : s.spot_lights) {
		Vec3 direction = math::normalize(toVec3(l.target) - toVec3(l.position));
		if (math::length(direction) < 0.5f) {
			direction = { 0.0f, -1.0f, 0.0f };
		}

		const float inner = std::min(l.inner_deg, l.outer_deg);
		const float outer = std::max(l.inner_deg, l.outer_deg);

		spot_data.push_back({
			{ l.position[0], l.position[1], l.position[2], 1.0f },
			{ direction.x, direction.y, direction.z, 0.0f },
			{ l.color[0], l.color[1], l.color[2], l.intensity },
			{ std::cos(math::radians(inner)), std::cos(math::radians(outer)), 0.0f, 0.0f },
		});
	}

	growStorageBuffer(point_buffer, 1, point_data.size() * sizeof(GpuPointLight));
	growStorageBuffer(spot_buffer, 2, spot_data.size() * sizeof(GpuSpotLight));

	const size_t point_capacity = size_t(point_buffer.size / sizeof(GpuPointLight));
	const size_t spot_capacity = size_t(spot_buffer.size / sizeof(GpuSpotLight));
	const size_t point_count = std::min(point_data.size(), point_capacity);
	const size_t spot_count = std::min(spot_data.size(), spot_capacity);

	writeBuffer(point_buffer, point_data.data(), point_count * sizeof(GpuPointLight));
	writeBuffer(spot_buffer, spot_data.data(), spot_count * sizeof(GpuSpotLight));

	const Mat4 view = buildView(s.camera);
	const Mat4 projection = buildProjection(s.camera, width / height);
	const Mat4 view_proj = projection * view;

	const Vec3 dir = directionFromAngles(s.directional.yaw_deg, s.directional.pitch_deg);
	const float dir_scale = s.directional.enabled ? s.directional.intensity : 0.0f;

	SceneData scene{};
	std::memcpy(scene.view_proj, view_proj.m, sizeof(scene.view_proj));
	scene.camera_position[0] = s.camera.position[0];
	scene.camera_position[1] = s.camera.position[1];
	scene.camera_position[2] = s.camera.position[2];
	scene.camera_position[3] = 1.0f;
	scene.ambient[0] = s.ambient_color[0] * s.ambient_intensity;
	scene.ambient[1] = s.ambient_color[1] * s.ambient_intensity;
	scene.ambient[2] = s.ambient_color[2] * s.ambient_intensity;
	scene.dir_direction[0] = dir.x;
	scene.dir_direction[1] = dir.y;
	scene.dir_direction[2] = dir.z;
	scene.dir_color[0] = s.directional.color[0] * dir_scale;
	scene.dir_color[1] = s.directional.color[1] * dir_scale;
	scene.dir_color[2] = s.directional.color[2] * dir_scale;
	scene.counts[0] = uint32_t(point_count);
	scene.counts[1] = uint32_t(spot_count);
	writeBuffer(scene_buffer, &scene, sizeof(scene));

	for (size_t i = 0; i < objects.size(); ++i) {
		const Object& o = objects[i];
		const Mat4 model = modelMatrix(int(i));
		const Mat4 normal = math::normalMatrix(model);

		ModelData data{};
		std::memcpy(data.model, model.m, sizeof(data.model));
		std::memcpy(data.normal_matrix, normal.m, sizeof(data.normal_matrix));
		data.albedo[0] = o.material.albedo[0];
		data.albedo[1] = o.material.albedo[1];
		data.albedo[2] = o.material.albedo[2];
		data.albedo[3] = 1.0f;
		data.specular[0] = o.material.specular[0];
		data.specular[1] = o.material.specular[1];
		data.specular[2] = o.material.specular[2];
		data.specular[3] = o.material.shininess;

		std::memcpy(static_cast<char*>(model_buffer.mapped) + size_t(model_stride) * i, &data, sizeof(data));
	}
	vmaFlushAllocation(context.allocator, model_buffer.allocation, 0, VK_WHOLE_SIZE);

	vkResetCommandBuffer(cb, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cb, &begin);

	VkClearValue clear_values[2];
	clear_values[0].color = { { 0.02f, 0.025f, 0.04f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cb, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = { 0.0f, 0.0f, width, height, 0.0f, 1.0f };
	const VkRect2D scissor = { { 0, 0 }, context.swapchain_extent };
	vkCmdSetViewport(cb, 0, 1, &viewport);
	vkCmdSetScissor(cb, 0, 1, &scissor);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cb, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cb, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_set, 0, nullptr);

	for (size_t i = 0; i < objects.size(); ++i) {
		const Object& o = objects[i];
		const Mesh& mesh = meshes[o.mesh];

		PushConstants push{};
		push.flags = (o.sampling.nontrivial ? FlagNonTrivialSampling : 0u) |
		             (o.sampling.box_blur ? FlagBoxBlur : 0u);
		push.time = s.effect_clock;
		push.amplitude = o.sampling.amplitude;
		push.blur_radius = o.sampling.box_blur ? o.sampling.blur_radius : 0;
		push.tint[0] = o.sampling.tint[0];
		push.tint[1] = o.sampling.tint[1];
		push.tint[2] = o.sampling.tint[2];
		push.tint[3] = 1.0f;

		vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

		const uint32_t dynamic_offset = uint32_t(model_stride * i);
		vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &textures[size_t(o.texture)].set, 1, &dynamic_offset);
		vkCmdDrawIndexed(cb, mesh.index_count, 1, mesh.first_index, mesh.vertex_offset, 0);
	}

	vkCmdEndRenderPass(cb);
	vkEndCommandBuffer(cb);
}

}
