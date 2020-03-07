#pragma once

#include "../Viewer/Viewer.h"

#include <glew.h>
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
	virtual ~GLFWViewer() override = default;
	virtual bool Init() override;
	virtual std::string help() override;
	virtual bool setupViewer() override;
	virtual bool render() override;
private:
	std::unique_ptr<GLFWwindow, glfwDeleter> m_window;
};