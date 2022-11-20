#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"

// cone measurements (in cm)
const float CENTIMETERS = 100;

class Cone{
    private:
        unsigned int coneVAO, coneVBO, coneEBO;
        unsigned int baseVAO, baseVBO, baseEBO;
    public:
        float base_width, base_height;
        float slant_length, top_circumference, base_circumference, largest_measurement;
        int num_sections, num_triangles;
        glm::vec3 position = glm::vec3(0.0f);
        Cone(float base_width, float base_height, float slant_length, float top_circumference, float base_circumference, int num_sections=100):
                base_width(base_width),
                base_height(base_height),
                slant_length(slant_length),
                top_circumference(top_circumference),
                base_circumference(base_circumference),
                num_sections(num_sections) {this->construct();}
        ~Cone();
    void construct_base();
    void construct_cone();
    void construct();

    void setPosition(float x, float y, float z){
        this->position = glm::vec3(x,y,z);
    }
    void setPosition(glm::vec3 pos){
        this->position = pos;
    };
    void Draw(Shader shader);
};