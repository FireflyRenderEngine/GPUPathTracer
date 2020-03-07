// GLFWViewer.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "GLFWViewer.h"

GLFWViewer::GLFWViewer()
{
	m_window = nullptr;
}

bool GLFWViewer::Init()
{
	// Initialise GLFW
	glewExperimental = true; // Needed for core profile
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
	m_window.reset(glfwCreateWindow(m_windowWidth, m_windowHeight, m_title.c_str(), NULL, NULL));
	if (!m_window.get()) 
	{
		//Error here: Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.
		glfwTerminate();
		return success;
	}

	glfwMakeContextCurrent(m_window.get()); // Initialize GLEW
	glewExperimental = true; // Needed in core profile
	if (glewInit() != GLEW_OK) 
	{
		//Error here: Failed to initialize GLEW
		return success;
	}
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
		glClear(GL_COLOR_BUFFER_BIT);

		// Draw nothing, see you in tutorial 2 !

		// Swap buffers
		glfwSwapBuffers(m_window.get());
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(m_window.get(), GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(m_window.get()) == 0);
	return false;
}