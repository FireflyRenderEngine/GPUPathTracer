#version 330 core
out vec4 FragColor;
in vec2 interpolatedVertexUV;
in vec3 interpolatedVertexNormals;

uniform vec3 geometryColor;

void main()
{
    FragColor = vec4(geometryColor.x, geometryColor.y, geometryColor.z, 1.0f);
}