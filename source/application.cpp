#include "application.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <math.h>

#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "cyl_math.hpp"

namespace application {

	namespace {

		auto& context = graphics::internal::context;

		struct Vertex {
			float position[3];
			float normal[3];
		};

		struct MarkerVertex {
			float position[3];
			float color[3];
		};
		static_assert(sizeof(Vertex) == 24 && sizeof(MarkerVertex) == 24);
		static_assert(offsetof(Vertex, normal) == 12 && offsetof(MarkerVertex, color) == 12);

		struct SceneUniforms {
			float view_projection[4][4];
			alignas(16) float camera_position[3];
			alignas(16) float directional_light_direction[3];
			alignas(16) float directional_light_color[3];
			alignas(16) float ambient_light[3];
			uint32_t point_light_count;
			uint32_t spot_light_count;
		};
		static_assert(sizeof(SceneUniforms) == 144);
		static_assert(offsetof(SceneUniforms, point_light_count) == 124);

		struct ModelUniform {
			float model[4][4];
			alignas(16) float albedo[3];
			alignas(16) float specular[3];
			float alpha;
		};
		static_assert(sizeof(ModelUniform) == 96);
		static_assert(offsetof(ModelUniform, alpha) == 92);

		struct PointLightGpu {
			alignas(16) float position[3];
			alignas(16) float color[3];
		};
		static_assert(sizeof(PointLightGpu) == 32);

		struct SpotLightGpu {
			alignas(16) float position[3];
			float cos_inner;
			alignas(16) float color[3];
			float cos_outer;
			alignas(16) float direction[3];
			float padding;
		};
		static_assert(sizeof(SpotLightGpu) == 48);

		constexpr uint32_t segment_count = 25;
		constexpr float cylinder_radius = 0.6f;
		constexpr float cylinder_half_height = 0.8f;
		constexpr uint32_t vertex_count = 2 * segment_count + 2;
		constexpr uint32_t index_count = segment_count * 12;
		float positions[vertex_count][3];
		uint32_t face_indices[index_count];
		Vertex vertices[index_count];
		uint32_t indices[index_count];

		constexpr float pi = 3.14159265f;
		constexpr float to_radians = pi / 180.0f;
		constexpr float near_plane = 0.1f;
		constexpr float far_plane = 100.0f;

		float camera_position[3] = { 0.0f, 0.0f, -4.0f };
		float camera_yaw_degrees = 0.0f;
		float camera_pitch_degrees = 0.0f;
		float fov_degrees = 60.0f;
		float camera_speed = 3.0f;

		bool animation_playing = true;
		float animation_speed = 1.0f;
		float orbit_radius = 1.0f;
		float wave_amplitude = 0.5f;
		int wave_count = 3;
		float spin_speed = 1.0f;
		float animation_time = 0.0f;
		double previous_time = 0.0;

		float model_position[3] = { 0.0f, 0.0f, 0.0f };
		float model_rotation_degrees[3] = { 0.0f, 0.0f, 0.0f };
		float model_scale[3] = { 1.0f, 1.0f, 1.0f };

		float material_albedo[3] = { 0.8f, 0.8f, 0.8f };
		float material_specular[3] = { 1.0f, 1.0f, 1.0f };
		float material_alpha = 32.0f;

		float ambient_color[3] = { 0.1f, 0.1f, 0.1f };
		float directional_direction[3] = { 0.3f, -0.6f, 0.8f };
		float directional_color[3] = { 1.0f, 1.0f, 1.0f };
		float directional_intensity = 0.7f;

		struct PointLightUi {
			float position[3];
			float color[3];
			float intensity;
		};

		struct SpotLightUi {
			float position[3];
			float color[3];
			float intensity;
			float direction[3];
			float inner_degrees;
			float outer_degrees;
		};

		constexpr int max_point_lights = 16;
		constexpr int max_spot_lights = 16;
		PointLightUi point_lights[max_point_lights] = {
			{ { 2.0f, 1.0f, -2.0f }, { 1.0f, 0.6f, 0.3f }, 10.0f },
		};
		int point_light_count = 1;
		SpotLightUi spot_lights[max_spot_lights] = {
			{ { -2.0f, 2.0f, -3.0f }, { 0.3f, 0.6f, 1.0f }, 30.0f, { 0.5f, -0.5f, 0.75f }, 15.0f, 25.0f },
		};
		int spot_light_count = 1;
		const PointLightUi new_point_light = { { 0.0f, 2.0f, -2.0f }, { 1.0f, 1.0f, 1.0f }, 10.0f };
		const SpotLightUi new_spot_light = { { 0.0f, 2.5f, -2.5f }, { 1.0f, 1.0f, 1.0f }, 30.0f,
			{ 0.0f, -0.7f, 0.7f }, 15.0f, 25.0f };

		bool show_light_markers = true;

		constexpr float marker_half_size = 0.07f;
		constexpr float cone_length = 1.5f;
		constexpr int cone_ring_segments = 24;
		constexpr int cone_spoke_step = 3;
		constexpr uint32_t cube_vertex_count = 36;
		constexpr uint32_t cone_line_vertex_count = 2 * cone_ring_segments +
			2 * (cone_ring_segments / cone_spoke_step) + 2;
		constexpr uint32_t max_marker_cube_vertices = (max_point_lights + max_spot_lights) * cube_vertex_count;
		constexpr uint32_t max_marker_line_vertices = max_spot_lights * cone_line_vertex_count;

		uint32_t marker_cube_vertices_used = 0;
		uint32_t marker_line_vertices_used = 0;

		VkShaderModule vk_vertex_shader;
		VkShaderModule vk_fragment_shader;
		VkShaderModule vk_marker_vertex_shader;
		VkShaderModule vk_marker_fragment_shader;

		VkDescriptorSetLayout vk_scene_set_layout;
		VkDescriptorSetLayout vk_model_set_layout;
		VkPipelineLayout vk_pipeline_layout;
		VkPipeline vk_pipeline;
		VkPipeline vk_marker_triangle_pipeline;
		VkPipeline vk_marker_line_pipeline;

		VkDescriptorPool vk_descriptor_pool;
		VkDescriptorSet vk_scene_set;
		VkDescriptorSet vk_model_set;

		VkBuffer vk_vertex_buffer;
		VmaAllocation vk_vertex_buffer_allocation;
		VkBuffer vk_index_buffer;
		VmaAllocation vk_index_buffer_allocation;
		VkBuffer vk_scene_buffer;
		VmaAllocation vk_scene_buffer_allocation;
		void* vk_scene_buffer_memory;
		VkBuffer vk_model_buffer;
		VmaAllocation vk_model_buffer_allocation;
		void* vk_model_buffer_memory;
		VkBuffer vk_point_lights_buffer;
		VmaAllocation vk_point_lights_buffer_allocation;
		void* vk_point_lights_buffer_memory;
		VkBuffer vk_spot_lights_buffer;
		VmaAllocation vk_spot_lights_buffer_allocation;
		void* vk_spot_lights_buffer_memory;
		VkBuffer vk_marker_cube_buffer;
		VmaAllocation vk_marker_cube_buffer_allocation;
		MarkerVertex* vk_marker_cube_buffer_memory;
		VkBuffer vk_marker_line_buffer;
		VmaAllocation vk_marker_line_buffer_allocation;
		MarkerVertex* vk_marker_line_buffer_memory;

		VkShaderModule loadShaderModule(const char path[]) {
			std::ifstream file(path, std::ios::binary | std::ios::ate);
			if (!file) {
				std::cerr << "Failed to open shader file " << path << '\n';
				return VK_NULL_HANDLE;
			}

			const size_t size = static_cast<size_t>(file.tellg());
			std::vector<uint32_t> buffer(size / sizeof(uint32_t));
			file.seekg(0);
			file.read(reinterpret_cast<char*>(buffer.data()), size);
			file.close();

			const VkShaderModuleCreateInfo info = {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = size,
				.pCode = buffer.data(),
			};

			VkShaderModule result;
			if (vkCreateShaderModule(context.device, &info, nullptr, &result) != VK_SUCCESS) {
				return VK_NULL_HANDLE;
			}

			return result;
		}

		bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
			VkBuffer& buffer, VmaAllocation& allocation, void** mapped) {
			const VkBufferCreateInfo buffer_info = {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			const VmaAllocationCreateInfo allocation_info = {
				.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
					VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO,
			};

			VmaAllocationInfo result_info;
			if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
				&buffer, &allocation, &result_info) != VK_SUCCESS) {
				return false;
			}

			*mapped = result_info.pMappedData;
			return true;
		}

		bool createPipeline(VkShaderModule vertex_shader, VkShaderModule fragment_shader,
			VkPrimitiveTopology topology, VkCullModeFlags cull_mode,
			VkPipeline& pipeline) {
			const VkPipelineShaderStageCreateInfo stage_infos[] = {
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = vertex_shader,
					.pName = "main",
				},
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
					.module = fragment_shader,
					.pName = "main",
				},
			};

			const VkVertexInputBindingDescription vertex_bindings[] = {
				{
					.binding = 0,
					.stride = sizeof(Vertex),
					.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
				},
			};

			const VkVertexInputAttributeDescription vertex_attributes[] = {
				{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = 0,
				},
				{
					.location = 1,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = 12,
				},
			};

			const VkPipelineVertexInputStateCreateInfo vertex_input_state = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
				.pVertexBindingDescriptions = vertex_bindings,
				.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
				.pVertexAttributeDescriptions = vertex_attributes,
			};

			const VkPipelineInputAssemblyStateCreateInfo assembly_state_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = topology,
			};

			const VkPipelineViewportStateCreateInfo viewport_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.scissorCount = 1,
			};

			const VkPipelineRasterizationStateCreateInfo raster_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = cull_mode,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.lineWidth = 1.0f,
			};

			const VkPipelineMultisampleStateCreateInfo sample_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			};

			const VkPipelineDepthStencilStateCreateInfo depth_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = true,
				.depthWriteEnable = true,
				.depthCompareOp = VK_COMPARE_OP_LESS,
			};

			const VkPipelineColorBlendAttachmentState attachment_info = {
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
					VK_COLOR_COMPONENT_G_BIT |
					VK_COLOR_COMPONENT_B_BIT |
					VK_COLOR_COMPONENT_A_BIT,
			};

			const VkPipelineColorBlendStateCreateInfo blend_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 1,
				.pAttachments = &attachment_info,
			};

			const VkDynamicState dynamic_states[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
			};

			const VkPipelineDynamicStateCreateInfo dynamic_state = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
				.pDynamicStates = dynamic_states,
			};

			const VkGraphicsPipelineCreateInfo pipeline_info = {
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = sizeof(stage_infos) / sizeof(stage_infos[0]),
				.pStages = stage_infos,
				.pVertexInputState = &vertex_input_state,
				.pInputAssemblyState = &assembly_state_info,
				.pViewportState = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState = &sample_info,
				.pDepthStencilState = &depth_info,
				.pColorBlendState = &blend_info,
				.pDynamicState = &dynamic_state,
				.layout = vk_pipeline_layout,
				.renderPass = context.render_pass,
				.subpass = 0,
			};

			return vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
				nullptr, &pipeline) == VK_SUCCESS;
		}

		cyl::Vec3 toVec3(const float v[3]) {
			return { v[0], v[1], v[2] };
		}

		cyl::Vec3 directionOf(const float v[3]) {
			const cyl::Vec3 d = toVec3(v);
			if (cyl::vlength(d) < 1e-4f) {
				return { 0.0f, -1.0f, 0.0f };
			}

			return cyl::vnormalize(d);
		}

		void buildCylinder() {
			const uint32_t n = segment_count;
			for (uint32_t i = 0; i < n; i++) {
				const float angle = 2.0f * pi * (float)i / (float)n;
				const float x = cylinder_radius * cosf(angle);
				const float z = cylinder_radius * sinf(angle);
				positions[i][0] = x;
				positions[i][1] = -cylinder_half_height;
				positions[i][2] = z;
				positions[n + i][0] = x;
				positions[n + i][1] = +cylinder_half_height;
				positions[n + i][2] = z;
			}

			positions[2 * n][0] = 0.0f;
			positions[2 * n][1] = -cylinder_half_height;
			positions[2 * n][2] = 0.0f;

			positions[2 * n + 1][0] = 0.0f;
			positions[2 * n + 1][1] = +cylinder_half_height;
			positions[2 * n + 1][2] = 0.0f;

			uint32_t k = 0;
			for (uint32_t i = 0; i < n; i++) {
				const uint32_t j = (i + 1) % n;
				const uint32_t bottom_i = i;
				const uint32_t bottom_j = j;
				const uint32_t top_i = n + i;
				const uint32_t top_j = n + j;

				face_indices[k++] = bottom_i;
				face_indices[k++] = top_i;
				face_indices[k++] = top_j;

				face_indices[k++] = bottom_i;
				face_indices[k++] = top_j;
				face_indices[k++] = bottom_j;

				face_indices[k++] = 2 * n;
				face_indices[k++] = bottom_i;
				face_indices[k++] = bottom_j;

				face_indices[k++] = 2 * n + 1;
				face_indices[k++] = top_j;
				face_indices[k++] = top_i;
			}
		}

		void generateVertices() {
			for (uint32_t t = 0; t < index_count / 3; t++) {
				const cyl::Vec3 a = toVec3(positions[face_indices[3 * t + 0]]);
				const cyl::Vec3 b = toVec3(positions[face_indices[3 * t + 1]]);
				const cyl::Vec3 c = toVec3(positions[face_indices[3 * t + 2]]);
				cyl::Vec3 normal = cyl::vnormalize(cyl::vcross(cyl::vsub(b, a), cyl::vsub(c, a)));
				const cyl::Vec3 center = cyl::vscale(cyl::vadd(cyl::vadd(a, b), c), 1.0f / 3.0f);
				if (cyl::vdot(normal, center) < 0.0f) {
					normal = cyl::vscale(normal, -1.0f);
				}

				const bool is_side = fabsf(normal.y) < 0.5f;
				for (uint32_t k = 0; k < 3; k++) {
					const uint32_t i = 3 * t + k;
					const float* p = positions[face_indices[i]];
					cyl::Vec3 vertex_normal = normal;
					if (is_side) {
						vertex_normal = cyl::vnormalize({ p[0], 0.0f, p[2] });
					}

					vertices[i] = {
						.position = { p[0], p[1], p[2] },
						.normal = { vertex_normal.x, vertex_normal.y, vertex_normal.z },
					};

					indices[i] = i;
				}
			}
		}

		cyl::Vec3 markerColor(const float color[3]) {
			const float m = fmaxf(color[0], fmaxf(color[1], color[2]));
			if (m < 1e-4f) {
				return { 1.0f, 1.0f, 1.0f };
			}

			return { color[0] / m, color[1] / m, color[2] / m };
		}

		void pushCube(MarkerVertex*& out, cyl::Vec3 center, cyl::Vec3 color) {
			cyl::Vec3 corners[8];
			for (int i = 0; i < 8; i++) {
				corners[i] = {
					center.x + ((i & 1) ? marker_half_size : -marker_half_size),
					center.y + ((i & 2) ? marker_half_size : -marker_half_size),
					center.z + ((i & 4) ? marker_half_size : -marker_half_size),
				};
			}

			static const int triangles[12][3] = {
				{ 0, 1, 3 }, { 0, 3, 2 },
				{ 4, 6, 7 }, { 4, 7, 5 },
				{ 0, 2, 6 }, { 0, 6, 4 },
				{ 1, 5, 7 }, { 1, 7, 3 },
				{ 0, 4, 5 }, { 0, 5, 1 },
				{ 2, 3, 7 }, { 2, 7, 6 },
			};

			for (int t = 0; t < 12; t++) {
				for (int k = 0; k < 3; k++) {
					const cyl::Vec3 p = corners[triangles[t][k]];
					*out++ = { { p.x, p.y, p.z }, { color.x, color.y, color.z } };
				}
			}
		}

		void pushLine(MarkerVertex*& out, cyl::Vec3 a, cyl::Vec3 b) {
			*out++ = { { a.x, a.y, a.z }, { 1.0f, 1.0f, 1.0f } };
			*out++ = { { b.x, b.y, b.z }, { 1.0f, 1.0f, 1.0f } };
		}

		void pushCone(MarkerVertex*& out, const SpotLightUi& light) {
			const cyl::Vec3 apex = toVec3(light.position);
			const cyl::Vec3 dir = directionOf(light.direction);
			const cyl::Vec3 helper = fabsf(dir.y) < 0.99f ? cyl::Vec3{ 0.0f, 1.0f, 0.0f }
			: cyl::Vec3{ 1.0f, 0.0f, 0.0f };
			const cyl::Vec3 right = cyl::vnormalize(cyl::vcross(helper, dir));
			const cyl::Vec3 up = cyl::vcross(dir, right);
			const cyl::Vec3 base_center = cyl::vadd(apex, cyl::vscale(dir, cone_length));
			const float radius = cone_length * tanf(light.outer_degrees * to_radians);
			pushLine(out, apex, base_center);
			cyl::Vec3 previous = {};
			for (int k = 0; k <= cone_ring_segments; k++) {
				const float angle = 2.0f * pi * (float)k / (float)cone_ring_segments;
				const cyl::Vec3 offset = cyl::vadd(cyl::vscale(right, cosf(angle) * radius),
					cyl::vscale(up, sinf(angle) * radius));
				const cyl::Vec3 point = cyl::vadd(base_center, offset);
				if (k > 0) {
					pushLine(out, previous, point);
					if ((k - 1) % cone_spoke_step == 0) {
						pushLine(out, apex, previous);
					}
				}

				previous = point;
			}
		}

		void buildMarkers() {
			marker_cube_vertices_used = 0;
			marker_line_vertices_used = 0;
			if (!show_light_markers) {
				return;
			}

			MarkerVertex* cubes = vk_marker_cube_buffer_memory;
			MarkerVertex* cubes_begin = cubes;
			for (int i = 0; i < point_light_count; i++) {
				pushCube(cubes, toVec3(point_lights[i].position), markerColor(point_lights[i].color));
			}

			for (int i = 0; i < spot_light_count; i++) {
				pushCube(cubes, toVec3(spot_lights[i].position), markerColor(spot_lights[i].color));
			}

			marker_cube_vertices_used = (uint32_t)(cubes - cubes_begin);
			MarkerVertex* lines = vk_marker_line_buffer_memory;
			MarkerVertex* lines_begin = lines;
			for (int i = 0; i < spot_light_count; i++) {
				pushCone(lines, spot_lights[i]);
			}

			marker_line_vertices_used = (uint32_t)(lines - lines_begin);
			vmaFlushAllocation(context.allocator, vk_marker_cube_buffer_allocation, 0, VK_WHOLE_SIZE);
			vmaFlushAllocation(context.allocator, vk_marker_line_buffer_allocation, 0, VK_WHOLE_SIZE);
		}

		void uploadScene() {
			const float aspect = context.swapchain_extent.height == 0
				? 1.0f
				: (float)context.swapchain_extent.width / (float)context.swapchain_extent.height;
			const cyl::Mat4 view = cyl::mul(
				cyl::rotate_x(-camera_pitch_degrees * to_radians),
				cyl::mul(cyl::rotate_y(-camera_yaw_degrees * to_radians),
					cyl::translate(-camera_position[0], -camera_position[1], -camera_position[2])));

			const cyl::Mat4 projection = cyl::perspective(fov_degrees * to_radians, aspect,
				near_plane, far_plane);
			const cyl::Mat4 view_projection = cyl::mul(projection, view);

			SceneUniforms scene = {};
			std::memcpy(scene.view_projection, view_projection.m, sizeof(view_projection.m));
			std::memcpy(scene.camera_position, camera_position, sizeof(camera_position));

			const cyl::Vec3 direction = directionOf(directional_direction);
			scene.directional_light_direction[0] = direction.x;
			scene.directional_light_direction[1] = direction.y;
			scene.directional_light_direction[2] = direction.z;
			for (int k = 0; k < 3; k++) {
				scene.directional_light_color[k] = directional_color[k] * directional_intensity;
				scene.ambient_light[k] = ambient_color[k];
			}

			scene.point_light_count = (uint32_t)point_light_count;
			scene.spot_light_count = (uint32_t)spot_light_count;
			std::memcpy(vk_scene_buffer_memory, &scene, sizeof(scene));
			vmaFlushAllocation(context.allocator, vk_scene_buffer_allocation, 0, VK_WHOLE_SIZE);
		}

		void uploadLights() {
			PointLightGpu gpu_points[max_point_lights] = {};
			for (int i = 0; i < point_light_count; i++) {
				for (int k = 0; k < 3; k++) {
					gpu_points[i].position[k] = point_lights[i].position[k];
					gpu_points[i].color[k] = point_lights[i].color[k] * point_lights[i].intensity;
				}
			}

			std::memcpy(vk_point_lights_buffer_memory, gpu_points, sizeof(gpu_points));
			vmaFlushAllocation(context.allocator, vk_point_lights_buffer_allocation, 0, VK_WHOLE_SIZE);

			SpotLightGpu gpu_spots[max_spot_lights] = {};
			for (int i = 0; i < spot_light_count; i++) {
				const cyl::Vec3 direction = directionOf(spot_lights[i].direction);
				gpu_spots[i].direction[0] = direction.x;
				gpu_spots[i].direction[1] = direction.y;
				gpu_spots[i].direction[2] = direction.z;
				gpu_spots[i].cos_inner = cosf(spot_lights[i].inner_degrees * to_radians);
				gpu_spots[i].cos_outer = cosf(spot_lights[i].outer_degrees * to_radians);
				for (int k = 0; k < 3; k++) {
					gpu_spots[i].position[k] = spot_lights[i].position[k];
					gpu_spots[i].color[k] = spot_lights[i].color[k] * spot_lights[i].intensity;
				}
			}

			std::memcpy(vk_spot_lights_buffer_memory, gpu_spots, sizeof(gpu_spots));
			vmaFlushAllocation(context.allocator, vk_spot_lights_buffer_allocation, 0, VK_WHOLE_SIZE);
		}

		void uploadModel() {
			const cyl::Mat4 orientation = cyl::mul(
				cyl::rotate_z(model_rotation_degrees[2] * to_radians),
				cyl::mul(cyl::rotate_y(model_rotation_degrees[1] * to_radians),
					cyl::rotate_x(model_rotation_degrees[0] * to_radians)));

			const float phase = animation_time;
			const float offset_x = orbit_radius * cosf(phase);
			const float offset_y = wave_amplitude * sinf((float)wave_count * phase);
			const float offset_z = orbit_radius * sinf(phase);
			const float spin_angle = phase * spin_speed;
			const cyl::Mat4 spin = cyl::mul(cyl::rotate_y(spin_angle),
				cyl::rotate_x(spin_angle * 0.5f));
			const cyl::Mat4 model = cyl::mul(
				cyl::translate(model_position[0] + offset_x,
					model_position[1] + offset_y,
					model_position[2] + offset_z),
				cyl::mul(orientation,
					cyl::mul(spin,
						cyl::scale(model_scale[0], model_scale[1], model_scale[2]))));

			ModelUniform uniform = {};
			std::memcpy(uniform.model, model.m, sizeof(model.m));
			std::memcpy(uniform.albedo, material_albedo, sizeof(material_albedo));
			std::memcpy(uniform.specular, material_specular, sizeof(material_specular));
			uniform.alpha = material_alpha;
			std::memcpy(vk_model_buffer_memory, &uniform, sizeof(uniform));
			vmaFlushAllocation(context.allocator, vk_model_buffer_allocation, 0, VK_WHOLE_SIZE);
		}

	}

	bool initialize() {
		buildCylinder();
		generateVertices();
		vk_vertex_shader = loadShaderModule("./shaders/shader.vert.spv");
		vk_fragment_shader = loadShaderModule("./shaders/shader.frag.spv");
		vk_marker_vertex_shader = loadShaderModule("./shaders/marker.vert.spv");
		vk_marker_fragment_shader = loadShaderModule("./shaders/marker.frag.spv");
		if (!vk_vertex_shader || !vk_fragment_shader ||
			!vk_marker_vertex_shader || !vk_marker_fragment_shader) {
			std::cerr << "Failed to load shaders\n";
			return false;
		}

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

		const VkDescriptorSetLayoutCreateInfo scene_set_layout = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
			.pBindings = scene_bindings,
		};

		if (vkCreateDescriptorSetLayout(context.device, &scene_set_layout, nullptr,
			&vk_scene_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create scene descriptor set layout\n";
			return false;
		}

		const VkDescriptorSetLayoutBinding model_bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			},
		};

		const VkDescriptorSetLayoutCreateInfo model_set_layout = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = sizeof(model_bindings) / sizeof(model_bindings[0]),
			.pBindings = model_bindings,
		};

		if (vkCreateDescriptorSetLayout(context.device, &model_set_layout, nullptr,
			&vk_model_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create model descriptor set layout\n";
			return false;
		}

		const VkDescriptorSetLayout set_layouts[] = { vk_scene_set_layout, vk_model_set_layout };

		const VkPipelineLayoutCreateInfo pipeline_layout = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
			.pSetLayouts = set_layouts,
		};

		if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
			&vk_pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create pipeline layout\n";
			return false;
		}

		if (!createPipeline(vk_vertex_shader, vk_fragment_shader,
			VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT, vk_pipeline) ||
			!createPipeline(vk_marker_vertex_shader, vk_marker_fragment_shader,
				VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_NONE,
				vk_marker_triangle_pipeline) ||
			!createPipeline(vk_marker_vertex_shader, vk_marker_fragment_shader,
				VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE,
				vk_marker_line_pipeline)) {
			std::cerr << "Failed to create graphics pipelines\n";
			return false;
		}

		void* mapped = nullptr;
		if (!createBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			vk_vertex_buffer, vk_vertex_buffer_allocation, &mapped)) {
			std::cerr << "Failed to create vertex buffer\n";
			return false;
		}

		std::memcpy(mapped, vertices, sizeof(vertices));
		vmaFlushAllocation(context.allocator, vk_vertex_buffer_allocation, 0, VK_WHOLE_SIZE);
		if (!createBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			vk_index_buffer, vk_index_buffer_allocation, &mapped)) {
			std::cerr << "Failed to create index buffer\n";
			return false;
		}

		std::memcpy(mapped, indices, sizeof(indices));
		vmaFlushAllocation(context.allocator, vk_index_buffer_allocation, 0, VK_WHOLE_SIZE);
		if (!createBuffer((sizeof(SceneUniforms) + 0xf) & ~0xf, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			vk_scene_buffer, vk_scene_buffer_allocation, &vk_scene_buffer_memory)) {
			std::cerr << "Failed to create scene uniform buffer\n";
			return false;
		}

		if (!createBuffer((sizeof(ModelUniform) + 0xf) & ~0xf, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			vk_model_buffer, vk_model_buffer_allocation, &vk_model_buffer_memory)) {
			std::cerr << "Failed to create model uniform buffer\n";
			return false;
		}

		if (!createBuffer(sizeof(PointLightGpu) * max_point_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			vk_point_lights_buffer, vk_point_lights_buffer_allocation,
			&vk_point_lights_buffer_memory)) {
			std::cerr << "Failed to create point lights storage buffer\n";
			return false;
		}

		if (!createBuffer(sizeof(SpotLightGpu) * max_spot_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			vk_spot_lights_buffer, vk_spot_lights_buffer_allocation,
			&vk_spot_lights_buffer_memory)) {
			std::cerr << "Failed to create spot lights storage buffer\n";
			return false;
		}

		if (!createBuffer(sizeof(MarkerVertex) * max_marker_cube_vertices,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			vk_marker_cube_buffer, vk_marker_cube_buffer_allocation, &mapped)) {
			std::cerr << "Failed to create marker cube buffer\n";
			return false;
		}

		vk_marker_cube_buffer_memory = static_cast<MarkerVertex*>(mapped);
		if (!createBuffer(sizeof(MarkerVertex) * max_marker_line_vertices,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			vk_marker_line_buffer, vk_marker_line_buffer_allocation, &mapped)) {
			std::cerr << "Failed to create marker line buffer\n";
			return false;
		}

		vk_marker_line_buffer_memory = static_cast<MarkerVertex*>(mapped);

		const VkDescriptorPoolSize descriptor_pool_sizes[] = {
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2,
			},
			{
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 2,
			},
		};

		const VkDescriptorPoolCreateInfo descriptor_pool = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 2,
			.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
			.pPoolSizes = descriptor_pool_sizes,
		};

		if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
			&vk_descriptor_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor pool\n";
			return false;
		}

		const VkDescriptorSetAllocateInfo scene_set_allocate = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = vk_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &vk_scene_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &scene_set_allocate,
			&vk_scene_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate scene descriptor set\n";
			return false;
		}

		const VkDescriptorSetAllocateInfo model_set_allocate = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = vk_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &vk_model_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &model_set_allocate,
			&vk_model_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate model descriptor set\n";
			return false;
		}

		const VkDescriptorBufferInfo scene_buffer_descriptor = {
			.buffer = vk_scene_buffer,
			.offset = 0,
			.range = sizeof(SceneUniforms),
		};

		const VkDescriptorBufferInfo point_lights_descriptor = {
			.buffer = vk_point_lights_buffer,
			.offset = 0,
			.range = sizeof(PointLightGpu) * max_point_lights,
		};

		const VkDescriptorBufferInfo spot_lights_descriptor = {
			.buffer = vk_spot_lights_buffer,
			.offset = 0,
			.range = sizeof(SpotLightGpu) * max_spot_lights,
		};

		const VkDescriptorBufferInfo model_buffer_descriptor = {
			.buffer = vk_model_buffer,
			.offset = 0,
			.range = sizeof(ModelUniform),
		};

		const VkWriteDescriptorSet descriptor_writes[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = vk_scene_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &scene_buffer_descriptor,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = vk_scene_set,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &point_lights_descriptor,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = vk_scene_set,
				.dstBinding = 2,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &spot_lights_descriptor,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = vk_model_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &model_buffer_descriptor,
			},
		};

		vkUpdateDescriptorSets(context.device,
			sizeof(descriptor_writes) / sizeof(descriptor_writes[0]),
			descriptor_writes, 0, nullptr);
		return true;
	}

	void shutdown() {
		vkQueueWaitIdle(context.graphics_queue);
		vkDestroyPipeline(context.device, vk_marker_line_pipeline, nullptr);
		vkDestroyPipeline(context.device, vk_marker_triangle_pipeline, nullptr);
		vkDestroyPipeline(context.device, vk_pipeline, nullptr);
		vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
		vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
		vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
		vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);
		vmaDestroyBuffer(context.allocator, vk_marker_line_buffer, vk_marker_line_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_marker_cube_buffer, vk_marker_cube_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_spot_lights_buffer, vk_spot_lights_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_point_lights_buffer, vk_point_lights_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_model_buffer, vk_model_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_scene_buffer, vk_scene_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
		vkDestroyShaderModule(context.device, vk_marker_fragment_shader, nullptr);
		vkDestroyShaderModule(context.device, vk_marker_vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, vk_fragment_shader, nullptr);
		vkDestroyShaderModule(context.device, vk_vertex_shader, nullptr);
	}

	void update(double time) {
		float delta = (float)(time - previous_time);
		previous_time = time;
		if (delta > 0.1f) {
			delta = 0.1f;
		}

		if (animation_playing) {
			animation_time += delta * animation_speed;
		}

		if (!ImGui::GetIO().WantTextInput) {
			const float yaw = camera_yaw_degrees * to_radians;
			const float pitch = camera_pitch_degrees * to_radians;
			const cyl::Vec3 forward = { sinf(yaw) * cosf(pitch), -sinf(pitch), cosf(yaw) * cosf(pitch) };
			const cyl::Vec3 right = { cosf(yaw), 0.0f, -sinf(yaw) };
			cyl::Vec3 move = { 0.0f, 0.0f, 0.0f };
			if (ImGui::IsKeyDown(ImGuiKey_W)) {
				move = cyl::vadd(move, forward);
			}

			if (ImGui::IsKeyDown(ImGuiKey_S)) {
				move = cyl::vsub(move, forward);
			}

			if (ImGui::IsKeyDown(ImGuiKey_D)) {
				move = cyl::vadd(move, right);
			}

			if (ImGui::IsKeyDown(ImGuiKey_A)) {
				move = cyl::vsub(move, right);
			}

			camera_position[0] += move.x * camera_speed * delta;
			camera_position[1] += move.y * camera_speed * delta;
			camera_position[2] += move.z * camera_speed * delta;
		}

		ImGui::Begin("Lighting");
		ImGui::SeparatorText("Camera (WASD to move)");
		ImGui::SliderFloat3("Camera position", camera_position, -10.0f, 10.0f);
		ImGui::SliderFloat("Yaw (deg)", &camera_yaw_degrees, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch (deg, + = down)", &camera_pitch_degrees, -89.0f, 89.0f);
		ImGui::SliderFloat("FOV (deg)", &fov_degrees, 20.0f, 120.0f);
		ImGui::SliderFloat("Camera speed", &camera_speed, 0.5f, 10.0f);
		ImGui::SeparatorText("Animation");
		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}

		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			animation_time = 0.0f;
		}

		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Orbit radius", &orbit_radius, 0.0f, 2.0f);
		ImGui::SliderFloat("Wave amplitude", &wave_amplitude, 0.0f, 1.5f);
		ImGui::SliderInt("Waves per orbit", &wave_count, 1, 8);
		ImGui::SliderFloat("Spin speed", &spin_speed, 0.0f, 5.0f);
		ImGui::SeparatorText("Model transform");
		ImGui::SliderFloat3("Position", model_position, -3.0f, 3.0f);
		ImGui::SliderFloat3("Rotation (deg)", model_rotation_degrees, -180.0f, 180.0f);
		ImGui::SliderFloat3("Scale", model_scale, 0.1f, 3.0f);
		ImGui::SeparatorText("Material");
		ImGui::ColorEdit3("Albedo", material_albedo);
		ImGui::ColorEdit3("Specular color", material_specular);
		ImGui::SliderFloat("Shininess", &material_alpha, 1.0f, 256.0f);
		ImGui::SeparatorText("Ambient light");
		ImGui::ColorEdit3("Ambient color", ambient_color);
		ImGui::SeparatorText("Directional light");
		ImGui::SliderFloat3("Direction", directional_direction, -1.0f, 1.0f);
		ImGui::ColorEdit3("Dir. color", directional_color);
		ImGui::SliderFloat("Dir. intensity", &directional_intensity, 0.0f, 5.0f);
		ImGui::SeparatorText("Point lights");
		ImGui::PushID("point_lights");
		if (ImGui::Button("Add point light") && point_light_count < max_point_lights) {
			point_lights[point_light_count++] = new_point_light;
		}

		int point_to_remove = -1;
		for (int i = 0; i < point_light_count; i++) {
			char label[32];
			std::snprintf(label, sizeof(label), "Point light %d", i + 1);
			ImGui::PushID(i);
			if (ImGui::TreeNode(label)) {
				ImGui::DragFloat3("Position", point_lights[i].position, 0.05f);
				ImGui::ColorEdit3("Color", point_lights[i].color);
				ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f, 1000.0f);
				if (ImGui::Button("Remove")) {
					point_to_remove = i;
				}

				ImGui::TreePop();
			}

			ImGui::PopID();
		}

		if (point_to_remove >= 0) {
			for (int i = point_to_remove; i + 1 < point_light_count; i++) {
				point_lights[i] = point_lights[i + 1];
			}

			point_light_count--;
		}

		ImGui::PopID();
		ImGui::SeparatorText("Spot lights");
		ImGui::PushID("spot_lights");
		if (ImGui::Button("Add spot light") && spot_light_count < max_spot_lights) {
			spot_lights[spot_light_count++] = new_spot_light;
		}

		int spot_to_remove = -1;
		for (int i = 0; i < spot_light_count; i++) {
			char label[32];
			std::snprintf(label, sizeof(label), "Spot light %d", i + 1);
			SpotLightUi& light = spot_lights[i];
			ImGui::PushID(i);
			if (ImGui::TreeNode(label)) {
				ImGui::DragFloat3("Position", light.position, 0.05f);
				ImGui::DragFloat3("Direction", light.direction, 0.01f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f);
				ImGui::SliderFloat("Inner angle (deg)", &light.inner_degrees, 0.0f, 85.0f);
				ImGui::SliderFloat("Outer angle (deg)", &light.outer_degrees, 1.0f, 85.0f);
				if (light.inner_degrees > light.outer_degrees) {
					light.inner_degrees = light.outer_degrees;
				}

				if (ImGui::Button("Remove")) {
					spot_to_remove = i;
				}

				ImGui::TreePop();
			}

			ImGui::PopID();
		}

		if (spot_to_remove >= 0) {
			for (int i = spot_to_remove; i + 1 < spot_light_count; i++) {
				spot_lights[i] = spot_lights[i + 1];
			}

			spot_light_count--;
		}

		ImGui::PopID();
		ImGui::SeparatorText("Visualization");
		ImGui::Checkbox("Show light markers", &show_light_markers);
		ImGui::End();
	}

	void render(const graphics::internal::FrameData& fd) {
		if (fd.command_buffer == VK_NULL_HANDLE) {
			return;
		}

		uploadScene();
		uploadLights();
		uploadModel();
		buildMarkers();
		vkResetCommandBuffer(fd.command_buffer, 0);

		const VkCommandBufferBeginInfo command_buffer_begin = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

		const VkClearValue clear_values[] = {
			{.color = {.float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } },
			{.depthStencil = { 1.0f, 0 } },
		};

		const VkRenderPassBeginInfo render_pass_begin = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = context.render_pass,
			.framebuffer = fd.framebuffer,
			.renderArea = {.extent = context.swapchain_extent },
			.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

		const VkViewport viewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(context.swapchain_extent.width),
			.height = static_cast<float>(context.swapchain_extent.height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		const VkRect2D scissor = { .extent = context.swapchain_extent };
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			vk_pipeline_layout, 0, 1, &vk_scene_set, 0, nullptr);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			vk_pipeline_layout, 1, 1, &vk_model_set, 0, nullptr);

		const VkDeviceSize vertex_buffer_offset = 0;
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &vertex_buffer_offset);
		vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
		if (marker_cube_vertices_used > 0) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				vk_marker_triangle_pipeline);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_marker_cube_buffer,
				&vertex_buffer_offset);
			vkCmdDraw(fd.command_buffer, marker_cube_vertices_used, 1, 0, 0);
		}

		if (marker_line_vertices_used > 0) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				vk_marker_line_pipeline);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_marker_line_buffer,
				&vertex_buffer_offset);
			vkCmdDraw(fd.command_buffer, marker_line_vertices_used, 1, 0, 0);
		}

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application