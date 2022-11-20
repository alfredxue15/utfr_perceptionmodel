LIBS = -lglut -lGLU -lGL -lglfw
INCLUDES = -Iinclude

all: include/glad.c cones.cpp simulation.cpp
	g++ -o simulation simulation.cpp cones.cpp include/glad.c $(INCLUDES) $(LIBS)

clean:
	-rm cones

