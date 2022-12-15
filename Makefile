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
SYCL_BUSINESS_CARD_RAYTRACER=sycl-business-card-raytracer${EXTENSION}
SYCL_MANDELBROT_USM=sycl-mandelbrot-usm${EXTENSION}
SYCL_MANDELBROT_BUFFER=sycl-mandelbrot-buffer${EXTENSION}

MAKEFILE=Makefile

all: ${SYCL_PATHTRACER} ${SYCL_BUSINESS_CARD_RAYTRACER} ${SYCL_MANDELBROT_USM} ${SYCL_MANDELBROT_BUFFER}

${SYCL_PATHTRACER}: sycl-pathtracer.cpp ${MAKEFILE}
	${CXX} -fsycl sycl-pathtracer.cpp -o ${SYCL_PATHTRACER}

${SYCL_BUSINESS_CARD_RAYTRACER}: sycl-business-card-raytracer.cpp ${MAKEFILE}
	${CXX} -fsycl sycl-business-card-raytracer.cpp -o ${SYCL_BUSINESS_CARD_RAYTRACER}

${SYCL_MANDELBROT_USM}: sycl-mandelbrot-usm.cpp ${MAKEFILE}
	${CXX} -fsycl sycl-mandelbrot-usm.cpp -o ${SYCL_MANDELBROT_USM}

${SYCL_MANDELBROT_BUFFER}: sycl-mandelbrot-buffer.cpp ${MAKEFILE}
	${CXX} -fsycl sycl-mandelbrot-buffer.cpp -o ${SYCL_MANDELBROT_BUFFER}
	
clean:
	${DEL} ${SYCL_PATHTRACER} ${SYCL_MANDELBROT_USM} ${SYCL_MANDELBROT_BUFFER}
