#pragma once

#include "../Viewer/Viewer.h"
#include <glad.h>
#include <glfw3.h>

#include <memory>

struct glfwDeleter
{
	void operator()(GLFWwindow* wnd)
	{
		glfwDestroyWindow(wnd);
	}
};

class GLFWViewer : public Viewer
{
public:
	GLFWViewer();
	GLFWViewer(std::shared_ptr<Scene> scene);
	virtual bool Init() override;
	virtual std::string help() override;
	virtual bool setupViewer() override;
	virtual bool render() override;

	virtual bool Create() override;
	virtual bool Draw() override;

	// This function is used to check & deal with any key press events
	void ProcessInput();

	virtual ~GLFWViewer() override 
	{
		glfwTerminate();
	}
private:
	std::unique_ptr<GLFWwindow, glfwDeleter> m_window;
};