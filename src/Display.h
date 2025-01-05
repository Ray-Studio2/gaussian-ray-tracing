#pragma once

#include <glad/glad.h>
#include <string>

class GLDisplay
{
public:
	GLDisplay();
	~GLDisplay();

	void display(const int32_t  screen_res_x,
				 const int32_t  screen_res_y,
				 const int32_t  framebuf_res_x,
				 const int32_t  framebuf_res_y,
				 const uint32_t pbo) const;

private:
	GLuint createGLProgram(const std::string& vert_source, const std::string& frag_source);
	GLuint createGLShader(const std::string& source, GLuint shader_type);
	GLint getGLUniformLocation(GLuint program, const std::string& name);

	GLuint   m_render_tex             = 0u;
	GLuint   m_program                = 0u;
	GLint    m_render_tex_uniform_loc = -1;
	GLuint   m_quad_vertex_buffer     = 0;

	static const std::string s_vert_source;
	static const std::string s_frag_source;
};