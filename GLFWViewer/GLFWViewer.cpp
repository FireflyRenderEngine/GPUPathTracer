// GLFWViewer.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "GLFWViewer.h"
#include "../Scene/Geometry.h"
#include "../Scene/TriangleMesh.h"
#include "../Scene/Cube.h"
#include "../Scene/Plane.h"
#include "../Scene/Sphere.h"
#include "../Scene/RasterCamera.h"
#include "gtc/type_ptr.hpp"
#include <random>
#include <sstream>
#include <fstream>

// The following set of functions are defined as overloads as the callbacks to different events triggered by GLFW
void FramebufferSizeCallback(GLFWwindow* window, int width, int height);

GLFWViewer::GLFWViewer() : Viewer(nullptr, nullptr)
{
	m_window = nullptr;
}

GLFWViewer::GLFWViewer(std::shared_ptr<Scene> Scene, std::shared_ptr<Film> film) : Viewer(Scene, film)
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
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
	return true;
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

	Create();

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
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		ProcessKeyboardInput();

		// Rendering commands for drawing to the screen
		Draw();

		// Swap buffers
		glfwSwapBuffers(m_window.get());
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(m_window.get(), GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(m_window.get()) == 0);
	return false;
}

void GLFWViewer::ProcessKeyboardInput()
{
	// process any key presses
	if (glfwGetKey(m_window.get(), GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(m_window.get(), true);

	if (glfwGetKey(m_window.get(), GLFW_KEY_W) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(FORWARD);
	if (glfwGetKey(m_window.get(), GLFW_KEY_S) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(BACKWARD);
	if (glfwGetKey(m_window.get(), GLFW_KEY_A) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(LEFT);
	if (glfwGetKey(m_window.get(), GLFW_KEY_D) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(RIGHT);
	if (glfwGetKey(m_window.get(), GLFW_KEY_Q) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(UP);
	if (glfwGetKey(m_window.get(), GLFW_KEY_E) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(DOWN);
	if (glfwGetKey(m_window.get(), GLFW_KEY_RIGHT) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(YAWRIGHT);
	if (glfwGetKey(m_window.get(), GLFW_KEY_LEFT) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(YAWLEFT);
	if (glfwGetKey(m_window.get(), GLFW_KEY_UP) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(PITCHUP);
	if (glfwGetKey(m_window.get(), GLFW_KEY_DOWN) == GLFW_PRESS)
		m_scene->m_rasterCamera->ProcessKeyboard(PITCHDOWN);
	// update 1st render camera
	if (!m_scene->m_cameras.empty())
	{
		m_scene->m_cameras[0] = m_scene->m_rasterCamera;
	}
}


std::string GetShaderCode(std::string filePath)
{
	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(filePath, std::ios::in);
	if (VertexShaderStream.is_open())
	{
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}
	else 
	{
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", filePath);
		getchar();
		return "";
	}
	return VertexShaderCode;
}

bool GLFWViewer::CompileFragmentShader(std::string fragmentShaderFilePath, unsigned int& fragmentShader) 
{
	bool success = false;

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	std::string fragmentShaderSource = GetShaderCode(fragmentShaderFilePath);

	char const* fragmentSourcePointer = fragmentShaderSource.c_str();
	glShaderSource(fragmentShader, 1, &fragmentSourcePointer, NULL);
	glCompileShader(fragmentShader);
	
	// Check Fragment Shader
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0)
	{
		std::vector<char> fragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(fragmentShader, InfoLogLength, NULL, &fragmentShaderErrorMessage[0]);
		return success;
	}

	success = true;
	return success;
}

bool GLFWViewer::CompileVertexShader(std::string vertexShaderFilePath, unsigned int& vertexShader) 
{
	bool success = false;

	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	std::string vertexShaderSource = GetShaderCode(vertexShaderFilePath);

	char const* vertexSourcePointer = vertexShaderSource.c_str();
	glShaderSource(vertexShader, 1, &vertexSourcePointer, NULL);
	glCompileShader(vertexShader);

	// Check Vertex Shader
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> vertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(vertexShader, InfoLogLength, NULL, &vertexShaderErrorMessage[0]);
		return success;
	}

	success = true;
	return success;
}

bool GLFWViewer::CreateShaderProgram(unsigned int& shaderProgram, unsigned int& vertexShader, unsigned int& fragmentShader)
{
	bool success = false;

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	GLint Result = GL_FALSE;
	int InfoLogLength = 0;
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &Result);
	std::vector<char> shaderProgramErrorMessage(InfoLogLength + 1);
	glGetShaderInfoLog(shaderProgram, InfoLogLength, NULL, &shaderProgramErrorMessage[0]);
	if (InfoLogLength > 0)
	{
		return success;
	}

	success = true;
	return success;
}

void GLFWViewer::DeleteShaders(unsigned int& vertexShader, unsigned int& fragmentShader)
{
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

bool GLFWViewer::Create() 
{
	bool success = false;

	// FORWARD PART: LOADING THE DATA FOR THE SCENE
	// Compile shaders for rendering the scene and create a shader program
	unsigned int vertexShader;
	std::string projectPath = SOLUTION_DIR;
	std::string vertexShaderFilePath = projectPath + R"(SceneResources\VertexShader.glsl)";
	if (!CompileVertexShader(vertexShaderFilePath, vertexShader)) 
	{
		return false;
	}

	// Compile Fragment Shader
	unsigned int fragmentShader;
	std::string fragmentShaderFilePath = projectPath + R"(SceneResources\FragmentShader.glsl)";
	if (!CompileFragmentShader(fragmentShaderFilePath, fragmentShader))
	{
		return false;
	}

	// Create a shader program
	if (!CreateShaderProgram(m_sceneShaderProgram, vertexShader, fragmentShader)) 
	{
		return false;
	}
	DeleteShaders(vertexShader, fragmentShader);

	// Seed the random number generator for creating random colors for geometries
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 1.0);

	// We will loop over the geometries and store the mesh data in GL Pointers
	for (int geometryIndex = 0; geometryIndex < m_scene->m_geometries.size(); geometryIndex++)
	{
		std::shared_ptr<Geometry> geometryPtr = m_scene->m_geometries[geometryIndex];
		
		unsigned int VAO;
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		// Set the VBO for the vertex buffer data
		unsigned int VBOVertexPos;
		glGenBuffers(1, &VBOVertexPos);
		glBindBuffer(GL_ARRAY_BUFFER, VBOVertexPos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * geometryPtr->m_vertices.size(), &(geometryPtr->m_vertices[0]), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		unsigned int VBOVertexUV;
		glGenBuffers(1, &VBOVertexUV);
		glBindBuffer(GL_ARRAY_BUFFER, VBOVertexUV);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * geometryPtr->m_uvs.size(), &(geometryPtr->m_uvs[0]), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);

		unsigned int VBOVertexNormals;
		glGenBuffers(1, &VBOVertexNormals);
		glBindBuffer(GL_ARRAY_BUFFER, VBOVertexNormals);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * geometryPtr->m_normals.size(), &(geometryPtr->m_normals[0]), GL_STATIC_DRAW);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(2);

		m_VAOS.push_back(VAO);
		
		// Set the color for the geometry to be visualized in the OpenGL Viewer
		m_randomColorPerGeometry.push_back(glm::vec3(dis(gen), dis(gen), dis(gen)));
	}

	// DEFERRED PART: LOADING THE DATA FOR THE RENDRER QUAD
	// Compile shaders for rendering the defered quad and create a shader program
	unsigned int deferredQuadVertexShader;
	std::string deferredQuadVertexShaderFilePath = projectPath + R"(SceneResources\DeferredQuadVertexShader.glsl)";
	if (!CompileVertexShader(deferredQuadVertexShaderFilePath, deferredQuadVertexShader))
	{
		return false;
	}

	// Compile Fragment Shader
	unsigned int deferredQuadFragmentShader;
	std::string deferredQuadFragmentShaderFilePath = projectPath + R"(SceneResources\DeferredQuadFragmentShader.glsl)";
	if (!CompileFragmentShader(deferredQuadFragmentShaderFilePath, deferredQuadFragmentShader))
	{
		return false;
	}

	// Create a shader program
	if (!CreateShaderProgram(m_deferredQuadShaderProgram, deferredQuadVertexShader, deferredQuadFragmentShader))
	{
		return false;
	}
	DeleteShaders(deferredQuadVertexShader, deferredQuadFragmentShader);

	// Load the Vertex Position, Normal, UV's for the quad
	glGenVertexArrays(1, &m_deferredQuadVAO);
	glBindVertexArray(m_deferredQuadVAO);

	// Set the VBO for the vertex buffer data
	unsigned int VBOVertexPos;
	glGenBuffers(1, &VBOVertexPos);
	glBindBuffer(GL_ARRAY_BUFFER, VBOVertexPos);
	std::vector<glm::vec3> vertexPosition = {glm::vec3(-1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(-1.0f, 0.0f, -1.0f),
											 glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, -1.0f)};
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * vertexPosition.size(), &(vertexPosition[0]), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	unsigned int VBOVertexUV;
	glGenBuffers(1, &VBOVertexUV);
	glBindBuffer(GL_ARRAY_BUFFER, VBOVertexUV);
	std::vector<glm::vec2> vertexNormal = {glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 0.0f) , glm::vec2(0.0f, 1.0f),
										   glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 0.0f), glm::vec2(1.0f, 1.0f)};
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * vertexNormal.size(), &(vertexNormal[0]), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);

	// Create framebuffer for deferred rendering. Attach texture to the framebuffer output
	glGenFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

	// Generate texture
	glGenTextures(1, &m_texColorBuffer);
	glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_scene->GetScreenWidth(), m_scene->GetScreenHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Attach it to currently bound framebuffer object
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texColorBuffer, 0);

	unsigned int rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_scene->GetScreenWidth(), m_scene->GetScreenHeight());
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
		return false;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	success = true;
	return success;
}

void GLFWViewer::UpdateProjectionMatrix() 
{
	int projectionMatrixLocation = glGetUniformLocation(m_sceneShaderProgram, "projectionMatrix");
	glm::mat4 projectionMatrix = m_scene->m_rasterCamera->GetProjectionMatrix();
	glUniformMatrix4fv(projectionMatrixLocation, 1, false, glm::value_ptr(projectionMatrix));
}

void GLFWViewer::SetGeometryColor(int geometryIndex)
{
	int geometryColorLocation = glGetUniformLocation(m_sceneShaderProgram, "geometryColor");
	glUniform3fv(geometryColorLocation, 1, &(m_randomColorPerGeometry[geometryIndex][0]));
}

void GLFWViewer::UpdateViewMatrix() 
{
	int viewMatrixLocation = glGetUniformLocation(m_sceneShaderProgram, "viewMatrix");
	glm::mat4 viewMatrix = m_scene->m_rasterCamera->GetViewMatrix();
	glUniformMatrix4fv(viewMatrixLocation, 1, false, glm::value_ptr(viewMatrix));
}

void GLFWViewer::SetGeometryModelMatrix(glm::mat4 modelMatrix) 
{
	int viewMatrixLocation = glGetUniformLocation(m_sceneShaderProgram, "modelMatrix");
	glUniformMatrix4fv(viewMatrixLocation, 1, false, glm::value_ptr(modelMatrix));
}

bool GLFWViewer::Draw()
{
	bool success = false;

	// Render the scene to the frame buffer attched texture
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // we're not using the stencil buffer now
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glUseProgram(m_sceneShaderProgram);
	glActiveTexture(GL_TEXTURE0);
	UpdateProjectionMatrix();
	UpdateViewMatrix();
	// Loop over the mesh and draw them
	for (int geometryIndex = 0; geometryIndex < m_VAOS.size(); geometryIndex++) 
	{
		glBindVertexArray(m_VAOS[geometryIndex]);
		// Bind the Model Matrix corrosponding to the current Geometry
		std::shared_ptr<Geometry> geometryPtr = m_scene->m_geometries[geometryIndex];
		SetGeometryModelMatrix(geometryPtr->m_modelMatrix);
		SetGeometryColor(geometryIndex);
		glDrawArrays(GL_TRIANGLES, 0, geometryPtr->m_triangleIndices.size());
	} 

	// Draw the rendered scene to the quad
	glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	if (glIsProgram(m_deferredQuadShaderProgram))
	{
		glUseProgram(m_deferredQuadShaderProgram);
		glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
		glBindVertexArray(m_deferredQuadVAO);
		glDisable(GL_DEPTH_TEST);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}
	
	success = true;
	return success;
}

void FramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}	
