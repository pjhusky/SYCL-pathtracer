// BUILD & RUN:
// clang++ -fsycl -O3 sycl-business-card-raytracer.cpp -o sycl-business-card-raytracer.exe & sycl-business-card-raytracer.exe > sycl-business-card-raytracer.ppm

// inspired by these two blog posts by Fabien Sanglard:
// https://fabiensanglard.net/rayTracing_back_of_business_card/
// https://fabiensanglard.net/revisiting_the_businesscard_raytracer/index.html

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <stdint.h>

#include <sycl/sycl.hpp>

static constexpr uint32_t DIM_X = 512u;
static constexpr uint32_t DIM_Y = 512u;

static constexpr uint32_t BPP = 3u;

#define INSTRINSIC 1

#if defined(INSTRINSIC)
#define POW sycl::pow<float>
#define CEIL sycl::ceil<float>
#define RSQRT(x) (sycl::rsqrt<float>(x))
#define SQRT(x) (sycl::sqrt<float>(x))
#define DIVIDE(x, y) ((x) / (y))
#else
#define POW powf
#define CEIL ceilf
#define RSQRT(x) (1.0f / sqrtf(x))
#define SQRT(x) (sqrtf((x)))
#define DIVIDE(x, y) ((x) / (y))
#endif

namespace {
    // Create an exception handler for asynchronous SYCL exceptions
    static auto exception_handler = []( sycl::exception_list e_list ) {
        for (std::exception_ptr const& e : e_list) {
            try {
                std::rethrow_exception( e );
            }
            catch (std::exception const& e) {
            #if _DEBUG
                std::cerr << "Failure" << std::endl;
            #endif
                std::terminate();
            }
        }
    };
}

struct v {
    float x, y, z;

    v( const float a, const float b, const float c ) {
        x = a;
        y = b;
        z = c;
    }

    inline v operator+( const v r ) const { return v( x + r.x, y + r.y, z + r.z ); }
    inline v operator*( const float r ) const { return v( x * r, y * r, z * r ); }
    inline float operator%( const v r ) const { return x * r.x + y * r.y + z * r.z; }
    inline v() {}
    inline v operator^( const v r ) const { return v( y * r.z - z * r.y, z * r.x - x * r.z, x * r.y - y * r.x ); }
    inline v operator!() const { return *this * RSQRT( *this % *this ); }
};

///*__device__*/ const int G[] = {247570, 280596, 280600, 249748, 18578, 18577, 231184, 16, 16}; // aek
//const int G[]={133022, 133266,133266, 133022, 254096, 131216, 131984, 131072, 258048,}; // Fab
// const int G[]={ // format: each int is one line of 3 characters (upside down) => 5-bit wide chars with 2-bit wide spaces between them
//   (0b00000<<16)|(0b00111<<8)|(0b10001), 
//   (0b01000<<16)|(0b00100<<8)|(0b10001), 
//   (0b01000<<16)|(0b00100<<8)|(0b10001), 
//   (0b01000<<16)|(0b00100<<8)|(0b10001), 
//   (0b01000<<16)|(0b00100<<8)|(0b10001), 
//   (0b01110<<16)|(0b00100<<8)|(0b11110), 
//   (0b01001<<16)|(0b00000<<8)|(0b10000), 
//   (0b01001<<16)|(0b00000<<8)|(0b10000), 
//   (0b01110<<16)|(0b00100<<8)|(0b10000),}; // pjh

const uint32_t G[] = { // format: each byte is one character line (upside down) => 4-bit wide chars with 2-bit wide spaces between them
  (0b01000u << 24u) | (0b000000u << 18u) | (0b000000u << 12u) | (0b001000u << 6u) | (0b000000u << 0u),
  (0b01000u << 24u) | (0b000000u << 18u) | (0b000000u << 12u) | (0b001000u << 6u) | (0b000000u << 0u),
  (0b01000u << 24u) | (0b000000u << 18u) | (0b000110u << 12u) | (0b001000u << 6u) | (0b000000u << 0u),
  (0b01000u << 24u) | (0b000000u << 18u) | (0b001000u << 12u) | (0b001000u << 6u) | (0b001001u << 0u),
  (0b01110u << 24u) | (0b001001u << 18u) | (0b001000u << 12u) | (0b001001u << 6u) | (0b001001u << 0u),
  (0b01001u << 24u) | (0b001001u << 18u) | (0b000110u << 12u) | (0b001010u << 6u) | (0b001001u << 0u),
  (0b01001u << 24u) | (0b001001u << 18u) | (0b000001u << 12u) | (0b001100u << 6u) | (0b000101u << 0u),
  (0b01001u << 24u) | (0b001001u << 18u) | (0b000001u << 12u) | (0b001010u << 6u) | (0b000001u << 0u),
  (0b01001u << 24u) | (0b000110u << 18u) | (0b001110u << 12u) | (0b001001u << 6u) | (0b000010u << 0u),
  (0b00000u << 24u) | (0b000000u << 18u) | (0b000000u << 12u) | (0b000000u << 6u) | (0b00100u << 0u),
}; //husky

float RANDOM( int& g_seed ) {
    g_seed = (214013 * g_seed + 2531011);
    return DIVIDE( (g_seed >> 16) & 0x7FFF, 66635.0f );
}

// The intersection test for line [o,v].
//  Return 2 if a hit was found (and also return distance t and bouncing ray n).
//  Return 0 if no hit was found but ray goes upward
//  Return 1 if no hit was found but ray goes downward
int TraceRay( const v origin, const v destination, float& t, v& normal ) {
    t = 1e9f;
    int m = 0;
    float p = DIVIDE( -origin.z, destination.z );
    if (.01f < p) {
        t = p;
        normal = v( 0, 0, 1 );
        m = 1;
    }

    // for (uint32_t k = 19; k--;)
    //   for (uint32_t j = 9; j--;)
    for (int32_t j = 0; j < _countof(G); j++) { 
        for (int32_t k = 0; k < sizeof(G[0])*8u; k++) { 
            if (G[j] & static_cast<uint32_t>(1u << static_cast<uint32_t>(k))) {
                //v p = origin + v(-k, 0, -j - 4); // upside down
                v p = origin + v( 14 - k, 0, -9 + j - 2 );
                float b = p % destination;
                float c = p % p - 1;
                float q = b * b - c;

                // Does the ray hit the sphere ?
                if (q > 0)
                {
                    float s = -b - SQRT( q );
                    // It does, compute the distance camera-sphere
                    if (s < t && s > 0.01f)
                    {
                        t = s;
                        normal = !(p + destination * t);
                        m = 2;
                    }
                }
            }
        }
    }
    return m;
}

v Sample( v origin, v destination, int& g_seed ) {
    float attenuation = 1.0f;
    v pixel_color( 0, 0, 0 );
    for (int r = 0; r < 4; r++, attenuation /= 2) {
        float t;
        v normal;

        int match = TraceRay( origin, destination, t, normal );
        if (!match)
        {
            // No sphere found and the ray goes upward: Generate a sky color
            return pixel_color + v( 0.7f, 0.6f, 1.0f ) * POW( 1 - destination.z, 4 ) * attenuation;
        }

        // A sphere was maybe hit.
        v intersection = origin + destination * t;
        v light_dir = !(v( 9 + RANDOM( g_seed ), 9 + RANDOM( g_seed ), 16 ) + intersection * -1);
        v half_vec = destination + normal * (normal % destination * -2);

        // Calculated the lambertian factor
        float lamb_f = light_dir % normal;

        // Calculate illumination factor (lambertian coefficient > 0 or in shadow)?
        if (lamb_f < 0 || TraceRay( intersection, light_dir, t, normal ))
        {
            lamb_f = 0;
        }

        if (match & 1)
        {
            // No sphere was hit and the ray was going downward: Generate a floor color
            intersection = intersection * .2f;
            v c = ((int)(CEIL( intersection.x ) + CEIL( intersection.y )) & 1 ? v( 3, 1, 1 ) : v( 3, 3, 3 )) * (lamb_f * .2f + .1f);
            return pixel_color + c * attenuation;
        }

        float color = POW( light_dir % half_vec * (lamb_f > 0), 99 );
        pixel_color = pixel_color + v( color, color, color ) * attenuation;
        // m == 2 A sphere was hit. Cast an ray bouncing from the sphere surface.
        // Attenuate color by 50% since it is bouncing (* .5)
        origin = intersection;
        destination = half_vec;
    }
    return pixel_color;
}

void GetColor( unsigned char* img, uint32_t gid ) {
    int g_seed = gid;
    const uint32_t threadId = static_cast<uint32_t>(gid);
    const int32_t x = threadId % DIM_X;
    const int32_t y = threadId / DIM_X;

    //v cam_dir = !v(-4, -16, 0); // more "orthogonal"
    //v cam_dir = !v(-6, -16, 0);
    v cam_dir = !v( -11, -16, 0 ); // more "slanted"
    //v cam_dir = !v(0, -20, 0);
    v cam_up = !(v( 0, 0, 1 ) ^ cam_dir) * .002f;
    v cam_right = !(cam_dir ^ cam_up) * .002f;// * ( static_cast<float>(DIM_X)/static_cast<float>(DIM_Y) );
    v eye_offset = (cam_up + cam_right) * -256 + cam_dir;
    v color( 13, 13, 13 );
    for (int r = 64; r--;)
        //for (int r = 256; r--;)
    {
        v delta = cam_up * (RANDOM( g_seed ) - .5f) * 99.0f + cam_right * (RANDOM( g_seed ) - 0.5f) * 99.0f;
        color = Sample(
            //v(17, 16, 8) + delta,
            v( 17.5f, 20, 8 ) + delta, // move cam to the left and slightly away from the text
            !(delta * -1 + (cam_up * (RANDOM( g_seed ) + x) + cam_right * (y + RANDOM( g_seed )) + eye_offset) * 16.0f), g_seed )
            * 3.5f +
            color;
    }
    img[DIM_X * y * BPP + x * BPP + 0] = color.x;
    img[DIM_X * y * BPP + x * BPP + 1] = color.y;
    img[DIM_X * y * BPP + x * BPP + 2] = color.z;
}

int main() {
    auto tstart = std::chrono::system_clock::now();		// take start time
    unsigned char* dev_bitmap;

    // The default device selector will select the most performant device.
    auto d_selector{ sycl::default_selector_v };
    try {
        sycl::queue q( d_selector, exception_handler );

        //!!! printf( "sycl shared malloc\n" );
        constexpr uint32_t numPixels = DIM_X * DIM_Y;
        dev_bitmap = sycl::malloc_shared<unsigned char>( numPixels * BPP, q );

        // Create the range object for the arrays.
        sycl::range<1> num_items{ numPixels };

        // Use parallel_for to run vector addition in parallel on device. This
        // executes the kernel.
        //    1st parameter is the number of work items.
        //    2nd parameter is the kernel, a lambda that specifies what to do per
        //    work item. the parameter of the lambda is the work item id.
        // SYCL supports unnamed lambda kernel by default.
        auto e = q.parallel_for(
            numPixels,
            [=]( auto gid ) {
                GetColor( dev_bitmap, gid );
            }
        );

        // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
        // the kernel asynchronously. Wait for the asynchronous call to complete.
        e.wait();

        auto tend = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count();

        //printf("P6 512 512 255 ");
        printf( "P6\n# duration=%fsec\n%d %d 255 ", duration, DIM_X, DIM_Y );

        char* c = reinterpret_cast<char*>(dev_bitmap);
        for ( uint32_t y = DIM_Y; y--; ) {
            for ( uint32_t x = DIM_X; x--; ) {
                c = reinterpret_cast<char*>(&dev_bitmap[y * DIM_X * BPP + x * BPP]);
                printf( "%c%c%c", c[0], c[1], c[2] );
                c += BPP;
            }
        }

        sycl::free( dev_bitmap, q );
    }
    catch (sycl::exception const& e) {
        std::cout << "An exception is caught: " << e.what() << std::endl;
        std::terminate();
    }

    return EXIT_SUCCESS;
}
