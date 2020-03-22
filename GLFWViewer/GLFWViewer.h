#pragma once

#include "../Viewer/Viewer.h"
#include <glad.h>
#include <glfw3.h>
#include "../glm-0.9.9.7/mat4x4.hpp"

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
	virtual void UpdateViewMatrix();
	virtual void SetGeometryModelMatrix(glm::mat4 modelMatrix);
	virtual void UpdateProjectionMatrix();


	// This function is used to check & deal with any key press events
	void ProcessKeyboardInput();
	void ProcessMouseInput();

	virtual ~GLFWViewer() override 
	{
		glfwTerminate();
	}
private:
	std::unique_ptr<GLFWwindow, glfwDeleter> m_window;

	// The set of VAO's assiciated with the vertex buffer data
	std::vector<unsigned int> m_VAOS;
	unsigned int m_shaderProgram;
};