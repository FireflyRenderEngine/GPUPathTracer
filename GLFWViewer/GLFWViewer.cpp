// GLFWViewer.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "GLFWViewer.h"

// The following set of functions are defined as overloads as the callbacks to different events triggered by GLFW
void FramebufferSizeCallback(GLFWwindow* window, int width, int height);

GLFWViewer::GLFWViewer()
{
	m_window = nullptr;
}

GLFWViewer::GLFWViewer(std::shared_ptr<Scene> Scene) : Viewer(Scene)
{
	m_window = nullptr;
}

bool GLFWViewer::Init()
{
	// Initialise GLFW
	if (!glfwInit())
	{
		// Error here: Failed to initialize GLFW
		return false;
	}
	return true;
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
}

std::string GLFWViewer::help()
{
	return std::string();
}

bool GLFWViewer::setupViewer()
{
	bool success = false;
	m_window.reset(glfwCreateWindow(m_scene->GetScreenWidth(), m_scene->GetScreenHeight(), m_title.c_str(), NULL, NULL));
	if (!m_window.get()) 
	{
		//Error here: Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.
		glfwTerminate();
		return success;
	}

	glfwMakeContextCurrent(m_window.get()); // Initialize GLEW

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		// Failed to initialize GLAD
		glfwTerminate();
		return success;
	}

	// Tell the OpenGL viewer the dimensions of the window
	// #Note: These dimensions are independet of the GLFW window dimensions. They are used to scale the 2D transformations between [-1, 1] on the screen 
	glViewport(0, 0, m_scene->GetScreenWidth(), m_scene->GetScreenHeight());

	// Register the callback functions with GLFW
	// These are user defined functions overloaded & to be called by GLFW
	glfwSetFramebufferSizeCallback(m_window.get(), FramebufferSizeCallback);

	success = true;
	return success;
}

bool GLFWViewer::render()
{
	bool success = false;
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(m_window.get(), GLFW_STICKY_KEYS, GL_TRUE);

	do 
	{
		// Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		ProcessInput();

		// Rendering commands for drawing to the screen

		// Swap buffers
		glfwSwapBuffers(m_window.get());
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(m_window.get(), GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(m_window.get()) == 0);
	return false;
}

void GLFWViewer::ProcessInput()
{
	// TODO: process any key presses
}

bool GLFWViewer::Create() {
	bool success = false;


	success = true;
	return success;
}

bool GLFWViewer::Draw() {
	bool success = false;


	success = true;
	return success;
}

void FramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}