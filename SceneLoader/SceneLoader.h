#pragma once

#include "../Scene/Scene.h"

#include <string>
#include <memory>


class SceneLoader
{
public:
	virtual bool LoadSceneFromFile(std::string filename) = 0;
	virtual std::shared_ptr<Scene> getScene() const
	{
		return m_scene;
	}
protected:
	std::shared_ptr<Scene> m_scene;
};