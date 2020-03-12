#pragma once

#include <string>
#include "../Scene/Scene.h"

class Viewer
{
public:
	Viewer() = default;
	Viewer(std::shared_ptr<Scene> scene)
		:m_scene(scene)
	{
	}
	virtual ~Viewer() = default;
	virtual bool Init() = 0;
	virtual std::string help() = 0;
	virtual bool setupViewer() = 0;
	virtual bool render() = 0;

	virtual bool Create() = 0;
	virtual bool Draw() = 0;
protected:
	std::string m_title{ "Firefly Engine" };
	std::shared_ptr<Scene> m_scene;
};