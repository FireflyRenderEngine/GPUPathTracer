#include "pch.h"
#include "Film.h"


Film::Film(std::string filename)
	:m_width(1024), m_height(768), m_pixels(new Rgba[m_width * m_height])
{
	m_imageRenderedStatus = false;
}

Film::Film(int width, int height)
	: m_width(width), m_height(height)
{
	m_pixels = new Rgba[width * height];
	m_imageRenderedStatus = false;
}

Film::~Film()
{
	delete m_pixels;
}

bool Film::saveAsEXR(std::string filename)
{
	// If the user does not provide a path to save the file then we will save it in the `Renders` folder in the project directory
	if (filename == "") {
		std::string projectPath = SOLUTION_DIR;
		std::string sceneFile = projectPath + R"(Renders\)";
		filename = sceneFile + R"(render.exr)";
	}

	RgbaOutputFile file(filename.c_str(), m_width, m_height);
	file.setFrameBuffer(m_pixels, 1, m_width);
	file.writePixels(m_height);
	return true;
}

void Film::SetImageRenderedStatus(bool status) 
{
	m_imageRenderedStatus = status;
}

bool Film::GetImageRenderedStatus() 
{
	return m_imageRenderedStatus;
}

void Film::SetFilm(Rgba* pixels) 
{
	m_pixels = pixels;
}

Rgba* Film::GetFilm() 
{
	return m_pixels;
}