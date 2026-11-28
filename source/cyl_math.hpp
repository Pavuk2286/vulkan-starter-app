#pragma once

#include <cmath>
#include <cstring>

namespace cyl {

    struct Mat4 {
        float m[4][4]; // m[column][row]
    };

    inline Mat4 identity() {
        Mat4 r;
        std::memset(&r, 0, sizeof(r));
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
        return r;
    }

    inline Mat4 mul(const Mat4& a, const Mat4& b) {
        Mat4 r;
        for (int c = 0; c < 4; c++) {
            for (int row = 0; row < 4; row++) {
                float s = 0.0f;
                for (int k = 0; k < 4; k++) s += a.m[k][row] * b.m[c][k];
                r.m[c][row] = s;
            }
        }
        return r;
    }

    inline Mat4 translate(float x, float y, float z) {
        Mat4 r = identity();
        r.m[3][0] = x; r.m[3][1] = y; r.m[3][2] = z;
        return r;
    }

    inline Mat4 scale(float x, float y, float z) {
        Mat4 r = identity();
        r.m[0][0] = x; r.m[1][1] = y; r.m[2][2] = z;
        return r;
    }

    inline Mat4 rotate_x(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[1][1] = c;  r.m[2][1] = -s;
        r.m[1][2] = s;  r.m[2][2] = c;
        return r;
    }

    inline Mat4 rotate_y(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0][0] = c;  r.m[2][0] = s;
        r.m[0][2] = -s; r.m[2][2] = c;
        return r;
    }

    inline Mat4 rotate_z(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0][0] = c;  r.m[1][0] = -s;
        r.m[0][1] = s;  r.m[1][1] = c;
        return r;
    }

    inline Mat4 perspective(float fov_y_radians, float aspect, float near_z, float far_z) {
        float t = std::tan(fov_y_radians * 0.5f);
        Mat4 r;
        std::memset(&r, 0, sizeof(r));
        r.m[0][0] = 1.0f / (aspect * t);
        r.m[1][1] = -1.0f / t;
        r.m[2][2] = far_z / (far_z - near_z);
        r.m[2][3] = 1.0f;
        r.m[3][2] = -far_z * near_z / (far_z - near_z);
        return r;
    }

    inline Mat4 orthographic(float left, float right, float bottom, float top,
        float near_z, float far_z) {
        Mat4 r = identity();
        r.m[0][0] = 2.0f / (right - left);
        r.m[1][1] = -2.0f / (top - bottom);
        r.m[2][2] = 1.0f / (far_z - near_z);
        r.m[3][0] = -(right + left) / (right - left);
        r.m[3][1] = (top + bottom) / (top - bottom);
        r.m[3][2] = -near_z / (far_z - near_z);
        return r;
    }

    struct Vec3 {
        float x, y, z;
    };

    inline Vec3 vadd(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
    inline Vec3 vsub(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
    inline Vec3 vscale(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
    inline float vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    inline float vlength(Vec3 a) { return std::sqrt(vdot(a, a)); }
    inline Vec3 vcross(Vec3 a, Vec3 b) {
        return { a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x };
    }

    inline Vec3 vnormalize(Vec3 a) {
        float len = vlength(a);
        if (len < 1e-6f) return { 0.0f, 0.0f, 0.0f };
        return vscale(a, 1.0f / len);
    }

} // namespace cyl