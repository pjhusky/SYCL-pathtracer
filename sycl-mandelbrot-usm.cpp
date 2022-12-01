// #define _USE_MATH_DEFINES
// #include <math.h>
// #include <iostream>
#include <algorithm>

#include <stdlib.h>     // pocketpt, single-source GLSL path tracer by Reinhold Preiner, 2020
#include <stdio.h>      // based on smallpt by Kevin Beason
#include <chrono>

#include <stdint.h>
#include <array>

#include <sycl/sycl.hpp>

// platform specific defines
#if defined(_WIN32) || defined(WIN32)
int fileopen(FILE **f, const char *filename) { return (int)fopen_s(f, filename, "w"); }
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
int fileopen(FILE **f, const char *filename) { *f = fopen(filename, "w"); return 0; }
#endif 

using uint3 = std::array<uint32_t, 3>;
using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

namespace {

    inline float clamp(const float x) { 
        return std::max( std::min( 1.0f, x ), 0.0f );
    }

    inline int toInt(float x) { 
        return int(sycl::pow(clamp(x), 1.0f / 2.2f) * 255.0f + 0.5f); 
    }		// performs gamma correction!

    // Create an exception handler for asynchronous SYCL exceptions
    static auto exception_handler = [](sycl::exception_list e_list) {
        for (std::exception_ptr const &e : e_list) {
            try {
            std::rethrow_exception(e);
            }
            catch (std::exception const &e) {
        #if _DEBUG
            std::cout << "Failure" << std::endl;
        #endif
            std::terminate();
            }
        }
    };
    
} // namespace

int main(int argc, char *argv[]) {
    
    //-- parse arguments
    const int32_t numIterations = argc>1 ? atoi(argv[1]) : 126;    // samples per pixel 
    const int32_t resy = argc>2 ? atoi(argv[2]) : 500;    // vertical pixel resolution
    const int32_t resx = resy; // horiziontal pixel resolution
    const uint32_t numPixels = resx * resy;
    
    const float fresx = static_cast< float >( resx );
    const float fresy = static_cast< float >( resy );
    
    auto tstart = std::chrono::system_clock::now();		// take start time

    // The default device selector will select the most performant device.
    auto d_selector{sycl::default_selector_v};

    try {
        sycl::queue q(d_selector, exception_handler);

        // Print out the device information used for the kernel code.
        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
        printf( "%d x %d @ %d iterations\n", resx, resy, numIterations );

        // Create arrays with "array_size" to store input and output data. Allocate
        // unified shared memory so that both CPU and device can access them.
        const uint32_t numPixels = resx * resy;
        vec4*   pMandel = sycl::malloc_shared<vec4>(numPixels, q);

        if ( pMandel == nullptr ) {
            std::cout << "Shared memory allocation failure.\n";
            return -1;
        }

        // vec2 centerComplex{ 0.0f, 0.0f };
        // vec2 halfDimComplex{ 2.0f, 2.0f };
        vec2 centerComplex{ -0.3f, 0.4f };
        vec2 halfDimComplex{ 1.0f, 1.0f };

        // Initialize radiances with 0.0f
        for (size_t i = 0; i < numPixels; i++) { pMandel[i] = vec4{ 0.0f, 0.0f, 0.0f, 0.0f }; }

        {
            // Create the range object for the arrays.
            sycl::range<1> num_items{numPixels};

            // Use parallel_for to run vector addition in parallel on device. This
            // executes the kernel.
            //    1st parameter is the number of work items.
            //    2nd parameter is the kernel, a lambda that specifies what to do per
            //    work item. the parameter of the lambda is the work item id.
            // SYCL supports unnamed lambda kernel by default.

            auto e = q.parallel_for(
                numPixels, 
                [=](auto gid) { 
                    
                    // determine what pixel we are calculating in this thread
                    // threadId is unique and is [0,pixelCount]
                    const uint32_t threadId = static_cast<uint32_t>(gid);
                    const uint32_t x = threadId % resx;
                    const uint32_t y = threadId / resx;
                    const uint32_t addr = x + ( resy - 1 - y ) * resx;
                    // const uint32_t addr = x + y * resx;
                    
                    const float fx = static_cast<float>( x );
                    const float fy = static_cast<float>( y );
                    
                    float cx = centerComplex[0] - halfDimComplex[0] + 2.0f * halfDimComplex[0] * ( fx / (fresx - 1.0f) );
                    float cy = centerComplex[1] - halfDimComplex[1] + 2.0f * halfDimComplex[1] * ( fy / (fresy - 1.0f) );
                    
                    const float sx = cx;
                    const float sy = cy;
                    int32_t iter = 0;
                    for ( ; iter < numIterations; iter++ ) {
                        const float cx2 = cx*cx;
                        const float cy2 = cy*cy;
                        if ( cx2 + cy2 >= 4.0f ) { break; }
                        float ncx = cx2 - cy2;
                        float ncy = 2.0f * cx * cy;
                        
                        cx = ncx + sx;
                        cy = ncy + sy;        
                    }
                    if ( iter == numIterations ) { iter = 0; }
                    const float fiter = static_cast<float>( iter ) / static_cast<float>( numIterations - 1 );
                    pMandel[ addr ] = vec4{ fiter, fiter, fiter, fiter };
                }                        
            );

            // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
            // the kernel asynchronously. Wait for the asynchronous call to complete.
            e.wait();            
            
        }

        auto tend = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count();

        printf( "render duration = %f sec\n", duration );

        //-- write inverse sensor image to file
        FILE *file;
        int err = fileopen(&file, "sycl-mandelbrot-usm.ppm");
        fprintf(file, "P3\n");
        fprintf(file, "# num iterations: %d\n", numIterations);
        fprintf(file, "# rendering time: %f s\n", duration);
        fprintf(file, "%d %d\n%d\n", resx, resy, 255);
        for (uint32_t i = 0; i < numPixels; i++) {
            fprintf(file, "%d %d %d ", toInt(pMandel[i][0]), toInt(pMandel[i][1]), toInt(pMandel[i][2]));
        }

        sycl::free(pMandel, q);

    } catch (sycl::exception const &e) {
        std::cout << "An exception is caught: " << e.what() << std::endl;
        std::terminate();
    }

}
