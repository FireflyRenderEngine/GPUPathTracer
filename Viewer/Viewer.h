#pragma once

#include <string>

class Viewer
{
public:
	Viewer() = default;
	Viewer(int windowWidth, int windowHeight)
		:m_windowWidth(windowWidth), m_windowHeight(windowHeight)
	{
	}
	virtual ~Viewer() = default;
	virtual bool Init() = 0;
	virtual std::string help() = 0;
	virtual bool setupViewer() = 0;
	virtual bool render() = 0;
protected:
	int m_windowWidth{ 1024 };
	int m_windowHeight{ 768 };
	std::string m_title{ "Firefly Engine" };
};