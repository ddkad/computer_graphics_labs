#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_frag_color;

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_proj;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	uvec4 counts;
} scene;

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	vec4 cone;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
	PointLight items[];
} point_lights;

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
	SpotLight items[];
} spot_lights;

layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;

layout(push_constant) uniform PushConstants {
	uint flags;
	float time;
	float amplitude;
	int blur_radius;
	vec4 tint;
} pc;

const uint FLAG_NONTRIVIAL_SAMPLING = 1u;
const uint FLAG_BOX_BLUR = 2u;

vec3 sampleAlbedo(vec2 uv) {
	if ((pc.flags & FLAG_BOX_BLUR) != 0u && pc.blur_radius > 0) {
		const vec2 texel = 1.0 / vec2(textureSize(albedo_texture, 0));
		const int r = pc.blur_radius;

		vec3 sum = vec3(0.0);
		for (int y = -r; y <= r; ++y) {
			for (int x = -r; x <= r; ++x) {
				sum += texture(albedo_texture, uv + vec2(float(x), float(y)) * texel).rgb;
			}
		}

		const float side = float(2 * r + 1);
		return sum / (side * side);
	}

	return texture(albedo_texture, uv).rgb;
}

vec3 nontrivialAlbedo(vec2 uv) {
	const float t = pc.time;

	const vec2 shift = pc.amplitude * 0.05 * vec2(cos(t), sin(1.3 * t));
	const vec3 red_copy = sampleAlbedo(uv + shift) * vec3(1.0, 0.25, 0.25);
	const vec3 cyan_copy = sampleAlbedo(uv - shift) * vec3(0.25, 1.0, 1.0);

	const float angle = 0.5 * t;
	const mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
	const float zoom = 1.5 + 0.5 * sin(t);
	const vec2 uv_layer = rotation * (uv - vec2(0.5)) * zoom + vec2(0.5);
	const vec3 layer = sampleAlbedo(uv_layer) * pc.tint.rgb;

	const float mix_factor = 0.35 + 0.25 * sin(t);
	return mix(red_copy + cyan_copy, layer, mix_factor);
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
	const float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	const vec3 H = normalize(L + V);
	const float n_dot_h = max(dot(N, H), 0.0);
	const float shininess = max(object.specular.w, 1.0);

	const vec3 diffuse = albedo * n_dot_l;
	const vec3 specular = object.specular.rgb * pow(n_dot_h, shininess);

	return (diffuse + specular) * radiance;
}

void main() {
	vec3 texel_color;
	if ((pc.flags & FLAG_NONTRIVIAL_SAMPLING) != 0u) {
		texel_color = nontrivialAlbedo(in_uv);
	} else {
		texel_color = sampleAlbedo(in_uv);
	}

	const vec3 albedo = texel_color * object.albedo.rgb;

	const vec3 N = normalize(in_world_normal);
	const vec3 V = normalize(scene.camera_position.xyz - in_world_position);

	vec3 color = scene.ambient.rgb * albedo;

	const vec3 dir_L = normalize(-scene.dir_direction.xyz);
	color += blinnPhong(N, V, dir_L, scene.dir_color.rgb, albedo);

	for (uint i = 0u; i < scene.counts.x; ++i) {
		const PointLight light = point_lights.items[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float dist_sq = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(dist_sq);

		const vec3 radiance = light.color.rgb * light.color.w / dist_sq;
		color += blinnPhong(N, V, L, radiance, albedo);
	}

	for (uint i = 0u; i < scene.counts.y; ++i) {
		const SpotLight light = spot_lights.items[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float dist_sq = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(dist_sq);

		const float cos_angle = dot(-L, normalize(light.direction.xyz));
		const float cos_inner = light.cone.x;
		const float cos_outer = light.cone.y;
		float edge = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
		edge = edge * edge * (3.0 - 2.0 * edge);

		const vec3 radiance = light.color.rgb * light.color.w * edge / dist_sq;
		color += blinnPhong(N, V, L, radiance, albedo);
	}

	out_frag_color = vec4(color, 1.0);
}
