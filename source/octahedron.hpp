#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct Vertex {
	float position[3];
	float color[3];
};

inline void makeOctahedron(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const float positions[6][3] = {
		{ +1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f },
		{ 0.0f, +1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
		{ 0.0f, 0.0f, +1.0f }, { 0.0f, 0.0f, -1.0f },
	};

	vertices.clear();
	for (const auto& p : positions) {
		vertices.push_back({
			{ p[0], p[1], p[2] },
			{ p[0] * 0.5f + 0.5f, p[1] * 0.5f + 0.5f, p[2] * 0.5f + 0.5f },
		});
	}

	indices.clear();
	for (int sx = -1; sx <= 1; sx += 2) {
		for (int sy = -1; sy <= 1; sy += 2) {
			for (int sz = -1; sz <= 1; sz += 2) {
				const uint16_t vx = sx > 0 ? 0 : 1;
				const uint16_t vy = sy > 0 ? 2 : 3;
				const uint16_t vz = sz > 0 ? 4 : 5;

				indices.push_back(vx);
				if (sx * sy * sz > 0) {
					indices.push_back(vy);
					indices.push_back(vz);
				} else {
					indices.push_back(vz);
					indices.push_back(vy);
				}
			}
		}
	}
}
