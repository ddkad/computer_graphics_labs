#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "math3d.hpp"

struct Vertex {
	float position[3];
	float normal[3];
};

inline void appendTriangle(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices,
                           math::Vec3 a, math::Vec3 b, math::Vec3 c) {
	const math::Vec3 n = math::normalize(math::cross(b - a, c - a));
	const uint16_t base = uint16_t(vertices.size());

	for (const math::Vec3& p : { a, b, c }) {
		vertices.push_back({ { p.x, p.y, p.z }, { n.x, n.y, n.z } });
	}

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

	vertices.clear();
	indices.clear();

	for (int sx = -1; sx <= 1; sx += 2) {
		for (int sy = -1; sy <= 1; sy += 2) {
			for (int sz = -1; sz <= 1; sz += 2) {
				const math::Vec3 vx = positions[sx > 0 ? 0 : 1];
				const math::Vec3 vy = positions[sy > 0 ? 2 : 3];
				const math::Vec3 vz = positions[sz > 0 ? 4 : 5];

				if (sx * sy * sz > 0) {
					appendTriangle(vertices, indices, vx, vy, vz);
				} else {
					appendTriangle(vertices, indices, vx, vz, vy);
				}
			}
		}
	}
}

inline void makePlane(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const math::Vec3 p0 = { -1.0f, 0.0f, +1.0f };
	const math::Vec3 p1 = { +1.0f, 0.0f, +1.0f };
	const math::Vec3 p2 = { +1.0f, 0.0f, -1.0f };
	const math::Vec3 p3 = { -1.0f, 0.0f, -1.0f };

	vertices.clear();
	indices.clear();

	appendTriangle(vertices, indices, p0, p1, p2);
	appendTriangle(vertices, indices, p0, p2, p3);
}
