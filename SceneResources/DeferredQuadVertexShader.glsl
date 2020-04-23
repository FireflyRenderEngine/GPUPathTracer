#version 330 core
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec2 vertexUV;

out vec2 outVectorUV;

void main()
{
    outVectorUV = vertexUV;
    gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0.0f, 1.0); 
}  