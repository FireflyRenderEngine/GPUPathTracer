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

#include <string>

class FILM_API Film
{
public:
	Film() = default;
	bool saveAsEXR(std::string filename);
private:
};