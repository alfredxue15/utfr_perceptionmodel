#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0f));
    Normal = mat3(transpose(inverse(model))) * aPos; // approximation, "ok" for high numbers of vertices
	gl_Position = projection * view * vec4(FragPos, 1.0f);
}