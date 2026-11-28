#version 450 core

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_uniforms;

struct PointLight {
    vec3 position;
    vec3 color;
};

struct SpotLight {
    vec3 position;
    float cos_inner;
    vec3 color;
    float cos_outer;
    vec3 direction;
    float padding;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 E) {
    float cos_theta = max(dot(n, l), 0.0);
    if (cos_theta == 0.0) {
        return vec3(0.0);
    }

    vec3 h = normalize(l + v);
    float cos_phi = max(dot(n, h), 0.0);
    float spec = pow(cos_phi, model_uniforms.alpha);

    return E * (cos_theta * model_uniforms.albedo + spec * model_uniforms.specular);
}

void main() {
    vec3 n = normalize(in_world_normal);
    vec3 v = normalize(scene_uniforms.camera_position - in_world_position);

    vec3 color = scene_uniforms.ambient_light * model_uniforms.albedo;

    vec3 l_dir = -normalize(scene_uniforms.directional_light_direction);
    color += blinn_phong(n, v, l_dir, scene_uniforms.directional_light_color);

    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 to_light = point_lights[i].position - in_world_position;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(dist2);
        color += blinn_phong(n, v, l, point_lights[i].color / dist2);
    }

    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3 to_light = spot_lights[i].position - in_world_position;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(dist2);

        float cos_gamma = dot(-l, normalize(spot_lights[i].direction));
        float range = max(spot_lights[i].cos_inner - spot_lights[i].cos_outer, 1e-4);
        float edge = clamp((cos_gamma - spot_lights[i].cos_outer) / range, 0.0, 1.0);

        color += blinn_phong(n, v, l, spot_lights[i].color * edge / dist2);
    }

    out_color = vec4(color, 1.0);
}