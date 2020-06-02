#version 440 core
layout (location = 0) in vec3 vertexPosition;

out vec2 outVectorUV;

void main()
{
    outVectorUV = (vertexPosition.xy + vec2(1, 1)) / 2.0;
    gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0.0f, 1.0); 
}