#version 330 core
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec2 vertexUV;
layout (location = 2) in vec3 vertexNormals;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelMatrix;

out vec2 outVertexUV;
out vec3 outVertexNormals;
void main()
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertexPosition.x, vertexPosition.y, vertexPosition.z, 1.0);
}