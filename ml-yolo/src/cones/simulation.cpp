#include <glad/glad.h>
#include <GLFW/glfw3.h>
//#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include "shader.h"
#include "camera.h"
#include "cones.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <random>
#include <ctime>

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
float frand(); // helper function to generate floats in [0,1]
float erand(); // helper function to generate exponential distributed random samples
void write_csv(std::string fname, std::vector<glm::vec4> bounding_boxes, std::vector<float> depths); // save results
glm::vec4 get_width_height(); // get bounding box from frame

// screen settings
const unsigned int SCR_WIDTH = 600;
const unsigned int SCR_HEIGHT = 800;
const float FOV = 67.4;  // questionable
const bool VISUAL = true;  // visualize which scenes are being simulated (does not really affect performance)
const bool CONTROL = false; // enable camera control (disables simulation)

// simulation parameters
//const unsigned int NUM_ITERS = 100000;  // how many images to take
const float INCHES_TO_CM = 2.54;
const float PI = glm::pi<float>();
const unsigned int MIN_RADIUS = 100; // minimum camera distance in centimeteres
const unsigned int RADIUS_INCR = 400; // scale of how much the exponential distributed term effects the distance
const float AVG_CAMERA_HEIGHT = 45*INCHES_TO_CM; // average height of camera off the ground
const float CAMERA_HEIGHT_DELTA = 15; // how much the camera height can deviate from the average
const float YAW_DELTA = 60; // how much the camera's looking direction can vary from looking straight on
const float PITCH_DELTA = 1; // how much (degrees) the camera can pitch up/down

// timing (for camera control)
float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool written = false;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;

// screen capture
typedef struct{
    u_char header[12];
    u_short width, height, header_2;
    u_char data[SCR_HEIGHT][SCR_WIDTH][3];  // RGB
} targa_file;
u_char default_header[12] = {0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
targa_file screen_img;

int main(int argc, char* argv[]){
    unsigned int NUM_ITERS = INT_MAX;
    if (!CONTROL){
        if (argc != 2){
            std::cout << "Please specify the number of simulation iterations you would like to run" << std::endl;
            return 1;
        } 
        NUM_ITERS = std::stoi(argv[1]);
        std::cout << "Running simulation for " << NUM_ITERS << " iterations." << std::endl;
    }

    
    glfwInit();
    GLFWwindow* window;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfw window creation
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "ConeSimulation", NULL, NULL);
    if (!window){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (VISUAL){
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    };

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    Shader coneShader("shaders/vertexshader.vs", "shaders/fragshader.fs");
    Shader screenShader("shaders/screenshader.vs", "shaders/screenshader.fs");
    screenShader.use();
    screenShader.setInt("screenTexture", 0);

    glm::mat4 projection = glm::perspective(glm::radians(FOV), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    coneShader.use();
    coneShader.setMat4("projection", projection);

    //Cone cone(30.f, 2.f, 45.f, 10.f, 65.f);
    //Cone cone(base_length, base_height, slant_len, top_circumference, base_circumference)
    Cone cone(9.*INCHES_TO_CM, 3.f, 12.1247680391*INCHES_TO_CM, 1.75*INCHES_TO_CM*PI, 5.7*INCHES_TO_CM*PI);

    coneShader.use();
    coneShader.setVec3("lightPos", glm::vec3(50, 50, 50));
    coneShader.setVec3("lightColor", glm::vec3(1.0, 1.0, 1.0));
    coneShader.setVec3("objectColor", glm::vec3(1.0, 0.5, 0));

    float screen_quad[] = {
        //position      texture coord
        -1.0f, -1.0f,   0.0f, 0.0f,
        -1.0f, 1.0f,    0.0f, 1.0f,
        1.0f, -1.0f,    1.0f, 0.0f,

        1.0f, -1.0f,   1.0f, 0.0f,
        1.0f, 1.0f,    1.0f, 1.0f,
        -1.0f, 1.0f,    0.0f, 1.0f,
    };

    unsigned int screenVBO, screenVAO;
    glGenVertexArrays(1, &screenVAO);
    glGenBuffers(1, &screenVBO);

    glBindVertexArray(screenVAO);
    glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(screen_quad), screen_quad, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); // positions
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2*sizeof(float))); // tex coords
    glEnableVertexAttribArray(0); // enable layout 0 and 1 in shader
    glEnableVertexAttribArray(1);

    // render to a frame buffer first so we can read the pixel data
    unsigned int FBO;
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    // colors buffer
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    // renderbuffer object for depth and stencil testing
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // allocate the memory
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // attach it to the FBO
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer is not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::vector<glm::vec4> bounding_boxes;
    std::vector<float> depths;
    // simulation loop
    for (int i=0; i<NUM_ITERS; i++){
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        if (VISUAL){
            glEnable(GL_DEPTH_TEST);
            float currentFrame = static_cast<float>(glfwGetTime());
            deltaTime = (currentFrame - lastFrame)/2;
            lastFrame = currentFrame;
        };
        processInput(window);
        if (glfwWindowShouldClose(window))
            break;


        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // randomly move camera
        glm::mat4 view;
        glm::vec3 cameraPos;
        if (!CONTROL){
            float dist = MIN_RADIUS + erand()*RADIUS_INCR;
            float angle = frand()*2*PI;
            float height = (frand()*2-1)*CAMERA_HEIGHT_DELTA;
            cameraPos = glm::vec3(glm::cos(angle)*dist, AVG_CAMERA_HEIGHT+height, glm::sin(angle)*dist)/CENTIMETERS;
            float pitch = glm::sin((frand()*2-1)*PITCH_DELTA*PI/180.);

            float angle_delta = angle + (frand()*2-1)*YAW_DELTA*PI/180.;
            glm::vec3 lookDir(-glm::cos(angle_delta), pitch, -glm::sin(angle_delta));

            view = glm::lookAt(cameraPos, cameraPos+glm::normalize(lookDir), glm::vec3(0.f,1.f,0.f));
            //std::cout << "pos: " << glm::to_string(cameraPos) << " look: " << glm::to_string(lookDir) << std::endl;
        }
        else{
            view = camera.GetViewMatrix();
            //std::cout << "pos: " << glm::to_string(camera.Position) << " look: " << glm::to_string(camera.Front) << std::endl;
        }

        coneShader.use();
        coneShader.setMat4("view", view);

        cone.Draw(coneShader);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);

        glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, &screen_img.data);
        bounding_boxes.push_back(get_width_height());
        depths.push_back(glm::length(cameraPos - cone.position));
        
        if (VISUAL){
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDisable(GL_DEPTH_TEST);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            
            screenShader.use();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureColorbuffer);

            glBindVertexArray(screenVAO);
            glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &screenVAO);
    glDeleteBuffers(1, &screenVBO);
    glfwTerminate();
    write_csv("simulation.csv", bounding_boxes, depths);
    return 0;
}

void write_csv(std::string fname, std::vector<glm::vec4> bounding_boxes, std::vector<float> depths){
    std::ofstream file;
    file.open(fname);
    auto bbox = bounding_boxes.begin();
    for (auto depth : depths){
        file << bbox->x << "," << bbox->y << "," << bbox->z << "," << bbox->w << "," << depth << "\n";
        bbox++;
    }
    file.close();
}

void processInput(GLFWwindow *window){
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (CONTROL){
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
            get_width_height();
    }
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (CONTROL){
        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

        lastX = xpos;
        lastY = ypos;

        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

void save_image(){
    if (!written){
        written = true;
        memcpy(screen_img.header, default_header, 12);
        screen_img.width = SCR_WIDTH;
        screen_img.height = SCR_HEIGHT;
        screen_img.header_2 = 0x2018; // who knows
        FILE* cc = fopen("data.tga", "wb");
        fwrite(&screen_img, 1, (18 + 3 * SCR_WIDTH * SCR_HEIGHT), cc);
        fclose(cc);
    }
}

glm::vec4 get_width_height(){
    //save_image();
    int first_x = 10000;
    int first_y = 10000;
    int last_x = -10000;
    int last_y = -10000;
    for (int y=0; y<SCR_HEIGHT; y++)
        for (int x=0; x<SCR_WIDTH; x++){
            bool non_zero = screen_img.data[y][x][1] > 0.1 && screen_img.data[y][x][0];
            if (non_zero){
                if (x<first_x) first_x = x;
                if (y<first_y) first_y = y;
                if (x>last_x) last_x = x;
                if (y>last_y) last_y = y;
            }
        }
    glm::vec4 result(first_x, first_y, last_x, last_y);
    if (CONTROL) std::cout << glm::to_string(result) << std::endl;
    return result;
}

float frand(){
    return (float)rand() / (float)RAND_MAX;
}

float erand(){
    static std::mt19937 mt(time(0));
    static std::exponential_distribution<float> exp_sampler(0.5);
    return exp_sampler(mt);
}