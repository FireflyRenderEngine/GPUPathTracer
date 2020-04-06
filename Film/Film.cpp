#include "pch.h"
#include "Film.h"


Film::Film(std::string filename)
	:m_width(1024), m_height(768), m_pixels(new Rgba[m_width * m_height])
{
}

Film::Film(int width, int height)
	: m_width(width), m_height(height)
{
	m_pixels = new Rgba[width * height];
}

Film::~Film()
{
	delete m_pixels;
}

bool Film::saveAsEXR(std::string filename)
{
	RgbaOutputFile file(filename.c_str(), m_width, m_height);
	file.setFrameBuffer(m_pixels, 1, m_width);
	file.writePixels(m_height);
	return true;
}
