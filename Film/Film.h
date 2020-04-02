#pragma once

#ifdef _USRDLL
#	ifdef FILM_EXPORTS
#		define FILM_API __declspec(dllexport)
#	else
#		define FILM_API __declspec(dllimport)
#	endif
#else
#	define FILM_API
#endif

#include <OpenEXR/IlmImf/ImfRgbaFile.h>
#include <OpenEXR/IlmImf/ImfArray.h>

#include <OpenEXR/IlmImf/ImfNamespace.h>

#include <string>

namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

class FILM_API Film
{
public:
	Film(int width, int height);
	~Film();
	bool saveAsEXR(std::string filename);
private:
	Rgba* m_pixels;
	int m_width;
	int m_height;
};