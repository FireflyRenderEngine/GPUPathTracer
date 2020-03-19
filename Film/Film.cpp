#include "pch.h"
#include "Film.h"

#include <OpenEXR/IlmImf/ImfRgbaFile.h>
#include <OpenEXR/IlmImf/ImfArray.h>

#include <OpenEXR/IlmImf/ImfNamespace.h>

namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

bool Film::saveAsEXR(std::string filename)
{
	RgbaOutputFile file(filename.c_str(), 1024, 768);
	/*file.setFrameBuffer(pixels, 1, width);
	file.writePixels(height);*/
	return false;
}
