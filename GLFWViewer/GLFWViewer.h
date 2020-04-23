#pragma once

#include "../Viewer/Viewer.h"
#include <glad.h>
#include <glfw3.h>
#include "mat4x4.hpp"

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
	GLFWViewer(std::shared_ptr<Scene> scene, std::shared_ptr<Film> film);
	virtual bool Init() override;
	virtual std::string help() override;
	virtual bool setupViewer() override;
	virtual bool render() override;

	virtual bool Create() override;
	bool CompileVertexShader(std::string vertexShaderFilePath, unsigned int& vertexShader);
	bool CompileFragmentShader(std::string fragmentShaderFilePath, unsigned int& fragmentShader);
	bool CreateShaderProgram(unsigned int& shaderProgram, unsigned int& vertexShader, unsigned int& fragmentShader);
	void DeleteShaders(unsigned int& vertexShader, unsigned int& fragmentShader);
	virtual bool Draw() override;
	virtual void UpdateViewMatrix();
	virtual void SetGeometryModelMatrix(glm::mat4 modelMatrix);
	virtual void UpdateProjectionMatrix();
	virtual void SetGeometryColor(int geometryIndex);

	// This function is used to check & deal with any key press events
	void ProcessKeyboardInput();

	virtual ~GLFWViewer() override 
	{
		glfwTerminate();
	}
private:
	std::unique_ptr<GLFWwindow, glfwDeleter> m_window;

	std::vector<unsigned int> m_VAOS; // The set of VAO's assiciated with the vertex buffer data
	std::vector<glm::vec3> m_randomColorPerGeometry;
	unsigned int m_sceneShaderProgram;
	
	unsigned int m_deferredQuadVAO;
	unsigned int m_deferredQuadShaderProgram;
	unsigned int m_framebuffer;
	unsigned int m_texColorBuffer;
};