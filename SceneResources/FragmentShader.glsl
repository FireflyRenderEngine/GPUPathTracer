#version 330 core
in vec2 outVertexUV;
in vec3 outVertexNormals;

uniform vec3 geometryColor; 

out vec4 FragColor;

void main()
{
    FragColor = vec4(geometryColor.x, geometryColor.y, geometryColor.z, 1.0f);
}