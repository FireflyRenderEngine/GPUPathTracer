#version 440 core
out vec4 FragColor;
  
in vec2 outVectorUV;

uniform sampler2D screenTexture;

void main()
{ 
    FragColor = texture(screenTexture, outVectorUV);
}