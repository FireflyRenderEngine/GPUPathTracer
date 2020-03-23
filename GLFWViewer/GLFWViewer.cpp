// GLFWViewer.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "GLFWViewer.h"
#include "../Scene/Geometry.h"
#include "../Scene/TriangleMesh.h"
#include "../Scene/RasterCamera.h"
#include "../glm-0.9.9.7/gtc/type_ptr.hpp"
#include <sstream>
#include <fstream>

// The following set of functions are defined as overloads as the callbacks to different events triggered by GLFW
void FramebufferSizeCallback(GLFWwindow* window, int width, int height);

GLFWViewer::GLFWViewer() : Viewer(nullptr)
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
}


std::string GetShaderCode(std::string filePath) {	
	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(filePath, std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}
	else {
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", filePath);
		getchar();
		return "";
	}
	return VertexShaderCode;
}

bool GLFWViewer::Create() {
	bool success = false;

	// Compile shaders and creata a shader program
	GLint Result = GL_FALSE;
	int InfoLogLength;
	// Compile Vertex Shader
	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	std::string vertexShaderFilePath = R"(C:\Users\rudra\Documents\Projects\FireflyRenderEngine\GPUPathTracer\SceneResources\VertexShader.glsl)";
	std::string vertexShaderSource = GetShaderCode(vertexShaderFilePath);

	char const* vertexSourcePointer = vertexShaderSource.c_str();
	glShaderSource(vertexShader, 1, &vertexSourcePointer, NULL);
	glCompileShader(vertexShader);

	// Check Vertex Shader
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> vertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(vertexShader, InfoLogLength, NULL, &vertexShaderErrorMessage[0]);
		return false;
	}

	// Compile Fragment Shader
	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	std::string fragmentShaderFilePath = R"(C:\Users\rudra\Documents\Projects\FireflyRenderEngine\GPUPathTracer\SceneResources\FragmentShader.glsl)";
	std::string fragmentShaderSource = GetShaderCode(fragmentShaderFilePath);

	char const* fragmentSourcePointer = fragmentShaderSource.c_str();
	glShaderSource(fragmentShader, 1, &fragmentSourcePointer, NULL);
	glCompileShader(fragmentShader);
	// Check Fragment Shader
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> fragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(fragmentShader, InfoLogLength, NULL, &fragmentShaderErrorMessage[0]);
		return false;
	}

	// Create a shader program
	m_shaderProgram = glCreateProgram();
	glAttachShader(m_shaderProgram, vertexShader);
	glAttachShader(m_shaderProgram, fragmentShader);
	glLinkProgram(m_shaderProgram);
	glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &Result);
	if (InfoLogLength > 0) {
		std::vector<char> shaderProgramErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(m_shaderProgram, InfoLogLength, NULL, &shaderProgramErrorMessage[0]);
		return false;
	}	

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// We will loop over the geometries and store the mesh data in GL Pointers
	for (int geometryIndex = 0; geometryIndex < m_scene->m_geometries.size(); geometryIndex++) {
		std::shared_ptr<Geometry> geometryPtr = m_scene->m_geometries[geometryIndex];
		if (geometryPtr->GetGeomtryType() == GeometryType::TRIANGLEMESH) {
			std::shared_ptr<TriangleMesh> triangleMeshGeometryPtr = std::static_pointer_cast<TriangleMesh>(geometryPtr);

			unsigned int VAO;
			glGenVertexArrays(1, &VAO);
			glBindVertexArray(VAO);

			// Set the VBO for the vertex buffer data
			unsigned int VBOVertexPos;
			glGenBuffers(1, &VBOVertexPos);
			glBindBuffer(GL_ARRAY_BUFFER, VBOVertexPos);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * triangleMeshGeometryPtr->m_vertices.size(), &(triangleMeshGeometryPtr->m_vertices[0]), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			unsigned int VBOVertexUV;
			glGenBuffers(1, &VBOVertexUV);
			glBindBuffer(GL_ARRAY_BUFFER, VBOVertexUV);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * triangleMeshGeometryPtr->m_uvs.size(), &(triangleMeshGeometryPtr->m_uvs[0]), GL_STATIC_DRAW);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);

			unsigned int VBOVertexNormals;
			glGenBuffers(1, &VBOVertexNormals);
			glBindBuffer(GL_ARRAY_BUFFER, VBOVertexNormals);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * triangleMeshGeometryPtr->m_normals.size(), &(triangleMeshGeometryPtr->m_normals[0]), GL_STATIC_DRAW);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(2);

			m_VAOS.push_back(VAO);
		}
	}

	success = true;
	return success;
}

void GLFWViewer::UpdateProjectionMatrix() {
	int projectionMatrixLocation = glGetUniformLocation(m_shaderProgram, "projectionMatrix");
	glm::mat4 projectionMatrix = m_scene->m_rasterCamera->GetProjectionMatrix();
	glUniformMatrix4fv(projectionMatrixLocation, 1, false, glm::value_ptr(projectionMatrix));
}

void GLFWViewer::UpdateViewMatrix() {
	int viewMatrixLocation = glGetUniformLocation(m_shaderProgram, "viewMatrix");
	glm::mat4 viewMatrix = m_scene->m_rasterCamera->GetViewMatrix();
	glUniformMatrix4fv(viewMatrixLocation, 1, false, glm::value_ptr(viewMatrix));
}


void GLFWViewer::SetGeometryModelMatrix(glm::mat4 modelMatrix) {
	int viewMatrixLocation = glGetUniformLocation(m_shaderProgram, "modelMatrix");
	glUniformMatrix4fv(viewMatrixLocation, 1, false, glm::value_ptr(modelMatrix));
}

bool GLFWViewer::Draw() {
	bool success = false;

	// Loop over the mesh 
	glUseProgram(m_shaderProgram);
	UpdateProjectionMatrix();
	UpdateViewMatrix();
	for (int geometryIndex = 0; geometryIndex < m_VAOS.size(); geometryIndex++) {
		glBindVertexArray(m_VAOS[geometryIndex]);
		// Bind the Model Matrix corrosponding to the current Geometry
		std::shared_ptr<Geometry> geometryPtr = m_scene->m_geometries[geometryIndex];
		if (geometryPtr->GetGeomtryType() == GeometryType::TRIANGLEMESH) {
			std::shared_ptr<TriangleMesh> triangleMeshGeometryPtr = std::static_pointer_cast<TriangleMesh>(geometryPtr);
			SetGeometryModelMatrix(triangleMeshGeometryPtr->m_modelMatrix);
			glDrawArrays(GL_TRIANGLES, 0, triangleMeshGeometryPtr->m_triangleIndices.size());
		}
	} 

	success = true;
	return success;
}

void FramebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}	
