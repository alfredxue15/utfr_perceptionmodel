#include "cones.h"

void Cone::construct_base(){
    float base_vertices[] = {
        0.5f, 0.5f, 0.5f, 
        0.5f, 0.5f, -.5f,
        0.5f, -.5f, 0.5f,
        0.5f, -.5f, -.5f,
        -.5f, 0.5f, 0.5f,
        -.5f, 0.5f, -.5f,
        -.5f, -.5f, 0.5f,
        -.5f, -.5f, -.5f,
    };
    int base_indices[] = {
        0,1,5, // +y face
        0,5,4,

        0,2,3, // +x face
        3,1,0, 

        0,2,6, // +z face
        0,6,4,

        2,3,7, // -y face
        7,2,6,

        6,7,5, // -x face
        5,6,4,

        1,3,7, // -z face
        7,1,5
    };

    glGenVertexArrays(1,&baseVAO);
    glGenBuffers(1, &baseVBO);
    glGenBuffers(1, &baseEBO);

    glBindVertexArray(baseVAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, baseEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(base_indices), base_indices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, baseVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(base_vertices), base_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); // for shader
    glEnableVertexAttribArray(0); // for shader
}

void Cone::construct_cone(){
    // construct the cone
    const float pi = glm::pi<float>();
    float base_radius = base_circumference/(2.*pi);
    float top_radius = top_circumference/(2.*pi);
    float height = sqrt(pow(slant_length,2) - pow(base_radius-top_radius, 2));
    this->largest_measurement = fmax(base_radius, height);

    float scaled_base_radius = base_radius/height;
    float scaled_top_radius = top_radius/height;
    float scaled_height = height/largest_measurement;
    
    this->num_triangles = num_sections*2;
    float circle[num_triangles*3];  // 3 floats for each vertex
    for (int i=0; i<(num_triangles)*3; i+=6){
        float angle = float(i)/(3.*num_triangles) * pi * 2;
        //float phase = 1./(NUM_SECTIONS)*pi*2/2;

        circle[i+0] = glm::cos(angle)*scaled_base_radius; // x1
        circle[i+1] = 0.; // y1
        circle[i+2] = glm::sin(angle)*scaled_base_radius; // z1

        circle[i+3] = glm::cos(angle)*scaled_top_radius; // x2
        circle[i+4] = scaled_height; // y2
        circle[i+5] = glm::sin(angle)*scaled_top_radius; // z2
    };

    int cone_indices[num_triangles*3]; // 3 indices per triangle
    int n = 0;  // which vertex we are using
    for (int i=0; i<num_triangles*3; i+=6){
        cone_indices[i+0] = (n+0) % (num_triangles); // base coord n
        cone_indices[i+1] = (n+1) % (num_triangles); // top coord n
        cone_indices[i+2] = (n+2) % (num_triangles); // base coord n+1

        cone_indices[i+3] = (n+2) % (num_triangles); // base_coord n+1
        cone_indices[i+4] = (n+1) % (num_triangles); // top_coord n
        cone_indices[i+5] = (n+3) % (num_triangles); // top_coord n+1
        n += 2;
    }; 

    glGenVertexArrays(1, &coneVAO);
    glGenBuffers(1, &coneVBO);
    glGenBuffers(1, &coneEBO);

    glBindVertexArray(coneVAO);
    glBindBuffer(GL_ARRAY_BUFFER, coneVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(circle), circle, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cone_indices), cone_indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); // for shader
    glEnableVertexAttribArray(0); // for shader
}

void Cone::Draw(Shader shader){
    shader.use();

    // draw cone
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = glm::scale(model, glm::vec3(largest_measurement/CENTIMETERS));
    shader.setMat4("model", model);
    glBindVertexArray(coneVAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneEBO);
    glDrawElements(GL_TRIANGLES, num_triangles*3, GL_UNSIGNED_INT, 0);

    // draw base
    model = glm::translate(glm::mat4(1.0f), position);
    model = glm::scale(model, glm::vec3(base_width, base_height, base_width)/CENTIMETERS);
    shader.setMat4("model", model);
    glBindVertexArray(baseVAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, baseEBO);
    glDrawElements(GL_TRIANGLES, 12*3, GL_UNSIGNED_INT, 0);
}

void Cone::construct(){
    this->construct_base();
    this->construct_cone();
}

Cone::~Cone(){
    glDeleteVertexArrays(1, &coneVAO);
    glDeleteVertexArrays(1, &baseVAO);
    glDeleteBuffers(1, &coneVBO);
    glDeleteBuffers(1, &coneEBO);
    glDeleteBuffers(1, &baseVBO);
    glDeleteBuffers(1, &baseEBO);
}