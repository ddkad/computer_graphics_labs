#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "math3d.hpp"

struct Vertex {
	float position[3];
	float normal[3];
	float uv[2];
};

struct Vec2 {
	float x;
	float y;
};

inline void appendTriangle(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices,
                           math::Vec3 a, math::Vec3 b, math::Vec3 c,
                           Vec2 uv_a, Vec2 uv_b, Vec2 uv_c) {
	const math::Vec3 n = math::normalize(math::cross(b - a, c - a));
	const uint16_t base = uint16_t(vertices.size());

	vertices.push_back({ { a.x, a.y, a.z }, { n.x, n.y, n.z }, { uv_a.x, uv_a.y } });
	vertices.push_back({ { b.x, b.y, b.z }, { n.x, n.y, n.z }, { uv_b.x, uv_b.y } });
	vertices.push_back({ { c.x, c.y, c.z }, { n.x, n.y, n.z }, { uv_c.x, uv_c.y } });

	indices.push_back(base);
	indices.push_back(uint16_t(base + 1));
	indices.push_back(uint16_t(base + 2));
}

inline void makeOctahedron(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const math::Vec3 positions[6] = {
		{ +1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f },
		{ 0.0f, +1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
		{ 0.0f, 0.0f, +1.0f }, { 0.0f, 0.0f, -1.0f },
	};

	const Vec2 uv_a = { 0.0f, 1.0f };
	const Vec2 uv_b = { 1.0f, 1.0f };
	const Vec2 uv_c = { 0.5f, 0.0f };

	vertices.clear();
	indices.clear();

	for (int sx = -1; sx <= 1; sx += 2) {
		for (int sy = -1; sy <= 1; sy += 2) {
			for (int sz = -1; sz <= 1; sz += 2) {
				const math::Vec3 vx = positions[sx > 0 ? 0 : 1];
				const math::Vec3 vy = positions[sy > 0 ? 2 : 3];
				const math::Vec3 vz = positions[sz > 0 ? 4 : 5];

				if (sx * sy * sz > 0) {
					appendTriangle(vertices, indices, vx, vy, vz, uv_a, uv_b, uv_c);
				} else {
					appendTriangle(vertices, indices, vx, vz, vy, uv_a, uv_b, uv_c);
				}
			}
		}
	}
}

inline void makePlane(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices, float tiling) {
	const math::Vec3 p0 = { -1.0f, 0.0f, +1.0f };
	const math::Vec3 p1 = { +1.0f, 0.0f, +1.0f };
	const math::Vec3 p2 = { +1.0f, 0.0f, -1.0f };
	const math::Vec3 p3 = { -1.0f, 0.0f, -1.0f };

	const Vec2 t0 = { 0.0f, tiling };
	const Vec2 t1 = { tiling, tiling };
	const Vec2 t2 = { tiling, 0.0f };
	const Vec2 t3 = { 0.0f, 0.0f };

	vertices.clear();
	indices.clear();

	appendTriangle(vertices, indices, p0, p1, p2, t0, t1, t2);
	appendTriangle(vertices, indices, p0, p2, p3, t0, t2, t3);
}
