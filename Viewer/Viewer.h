#pragma once

#include <string>
#include "../Film/Film.h"
#include "../Scene/Scene.h"

class Viewer
{
public:
	Viewer() = default;
	Viewer(std::shared_ptr<Scene> scene, std::shared_ptr<Film> film)
		:m_scene(scene), m_film(film)
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
	std::shared_ptr<Film> m_film;
};