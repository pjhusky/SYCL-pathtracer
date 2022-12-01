DEBUG ?= 1
ifeq (DEBUG, 1)
    CFLAGS =-g3 -gdwarf2 -DDEBUG
else
    CFLAGS=-DNDEBUG -O3
endif

DEL=del /Q /S 

CXX = clang++ $(CFLAGS)
CC = gcc $(CFLAGS)

EXTENSION=.exe

SYCL_PATHTRACER=sycl-pathtracer${EXTENSION}
SYCL_MANDELBROT=sycl-mandelbrot${EXTENSION}

MAKEFILE=Makefile

all: ${SYCL_PATHTRACER} ${SYCL_MANDELBROT}

${SYCL_PATHTRACER}: sycl-pathtracer.cpp ${MAKEFILE}
	${CXX} -fsycl sycl-pathtracer.cpp  -o ${SYCL_PATHTRACER}

${SYCL_MANDELBROT}: sycl-mandelbrot.cpp ${MAKEFILE}
	${CXX} -fsycl sycl-mandelbrot.cpp -o ${SYCL_MANDELBROT}
	
clean:
	${DEL} ${SYCL_PATHTRACER} ${SYCL_MANDELBROT}