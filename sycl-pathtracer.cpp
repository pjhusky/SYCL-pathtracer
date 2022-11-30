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

using uint3 = std::array<uint32_t, 3>;
using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

// platform specific defines
#if defined(_WIN32) || defined(WIN32)
int fileopen(FILE **f, const char *filename) { return (int)fopen_s(f, filename, "w"); }
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
int fileopen(FILE **f, const char *filename) { *f = fopen(filename, "w"); return 0; }
#endif 

namespace {
    constexpr float pi = 3.141592653589793f;
    
    struct RandomSeed {
        uint32_t s1,s2;
    };

    struct Ray { 
        vec3 o; 
        vec3 d; 
    };

    struct Sphere { 
        vec4 geo; 
        vec4 e; 
        vec4 c; 
    };

    constexpr Sphere spheres[] = {  // center.xyz, radius  |  emmission.xyz, 0  |  color.rgb, refltype     
        vec4{ 1e5f - 2.6f, 0.0f, 0.0f,  1e5f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.850f, 0.250f, 0.250f, 1.0f }, // Left (1 .. DIFFUSE)
        vec4{ 1e5f + 2.6f, 0.0f, 0.0f,  1e5f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.250f, 0.350f, 0.850f, 1.0f }, // Right
        vec4{ 0.0f, 1e5f + 2.0f, 0.0f,  1e5f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.750f, 0.750f, 0.750f, 1.0f }, // Top
        vec4{ 0.0f,-1e5f - 2.0f, 0.0f,  1e5f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.750f, 0.750f, 0.750f, 1.0f }, // Bottom
        vec4{ 0.0f, 0.0f, -1e5f - 2.8f, 1e5f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.850f, 0.850f, 0.250f, 1.0f }, // Back 
        vec4{ 0.0f, 0.0f, 1e5f + 7.9f,  1e5f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.100f, 0.700f, 0.700f, 1.0f }, // Front
        vec4{ -1.3f, -1.2f, -1.3f,      0.8f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.999f, 0.999f, 0.999f, 2.0f }, // 2 .. REFLECTIVE
        vec4{ 1.3f, -1.2f, -0.2f,       0.8f },  vec4{   0.0f,  0.0f,  0.0f, 0.0f },  vec4{ 0.999f, 0.999f, 0.999f, 3.0f }, // 3 .. REFRACTIVE
        vec4{ 0.0f, 2.0f*0.8f, 0.0f,    0.2f },  vec4{ 100.0f,100.0f,100.0f, 0.0f },  vec4{ 0.000f, 0.000f, 0.000f, 1.0f }, // Light
    };

    inline float clamp(const float x) { 
        //return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; 
        return std::max( std::min( 1.0f, x ), 0.0f );
    }

    inline float clamp(const float x, const float minVal, const float maxVal) { 
        return std::max( std::min( maxVal, x ), minVal ); 
    }

    inline int toInt(float x) { return int(sycl::pow(clamp(x), 1.0f / 2.2f) * 255.0f + 0.5f); }		// performs gamma correction!

    template<typename val_T, size_t numElements>
    inline std::array<val_T, 3> to_vec3( const std::array<val_T, numElements>& vec ) {
        assert( numElements == 4 );
        return std::array<val_T, 3>{ vec[0], vec[1], vec[2] };
    }

    template<typename val_T, size_t numElements> 
    inline std::array<val_T, 4> to_vec4( const std::array<val_T, numElements>& vec ) {
        assert( numElements == 3 );
        return std::array<val_T, 4>{ vec[0], vec[1], vec[2], 0.0f };
    }

    template<typename val_T, size_t numElements>
    inline std::array<val_T, numElements> add( const std::array<val_T, numElements>& lhs,
                                               const std::array<val_T, numElements>& rhs ) {
        
        std::array<val_T, numElements> retVal;
        for ( size_t i = 0; i < numElements; i++ ) {
            retVal[ i ] = lhs[ i ] + rhs[ i ];
        }
        return retVal;
    }

    template<typename val_T, size_t numElements>
    inline std::array<val_T, numElements> sub( const std::array<val_T, numElements>& lhs,
                                               const std::array<val_T, numElements>& rhs ) {
        
        std::array<val_T, numElements> retVal;
        for ( size_t i = 0; i < numElements; i++ ) {
            retVal[ i ] = lhs[ i ] - rhs[ i ];
        }
        return retVal;
    }

    template<typename val_T, size_t numElements>
    inline std::array<val_T, numElements> mul( const std::array<val_T, numElements>& lhs,
                                               const val_T factor ) {
        
        std::array<val_T, numElements> retVal;
        for ( size_t i = 0; i < numElements; i++ ) {
            retVal[ i ] = lhs[ i ] * factor;
        }
        return retVal;
    }

    template<typename val_T, size_t numElements>
    inline std::array<val_T, numElements> mul( const std::array<val_T, numElements>& lhs,
                                               const std::array<val_T, numElements>& rhs ) {
        
        std::array<val_T, numElements> retVal;
        for ( size_t i = 0; i < numElements; i++ ) {
            retVal[ i ] = lhs[ i ] * rhs[ i ];
        }
        return retVal;
    }

    template<typename val_T, size_t numElements>
    inline val_T dot( const std::array<val_T, numElements>& lhs,
                      const std::array<val_T, numElements>& rhs ) {
        
        val_T accum = val_T{0};
        for ( size_t i = 0; i < numElements; i++ ) {
            accum += lhs[ i ] * rhs[ i ];
        }
        return accum;
    }

    template<typename val_T, size_t numElements>
    inline std::array<val_T, numElements> normalize( const std::array<val_T, numElements>& vec ) {
        return mul( vec, val_T{ 1 } / sycl::sqrt( dot( vec, vec ) ) );
    }

    template<typename val_T>
    inline std::array<val_T, 3> cross( const std::array<val_T, 3>& lhs,
                                       const std::array<val_T, 3>& rhs ) {
        
        std::array<val_T, 3> retVal;
        retVal[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
        retVal[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
        retVal[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
        
        return retVal;
    }

    vec3 reflect( vec3 inVec, vec3 normal ) {
        return sub( inVec, mul( normal, 2.0f * dot( inVec, normal ) ) );
    }

    bool intersect( const Ray& r, 
                    const Sphere *const pSpheres, 
                    uint32_t& id, 
                    vec3& x, 
                    vec3& n ) { // intersect ray with scene
                    
        constexpr float inf = 1e20f;
        constexpr float eps = 1e-4f;
        
        float d, t = inf;
        
        for (uint32_t i = 0; i < _countof(spheres); i++) { 
            const Sphere& s = pSpheres[i];                  // perform intersection test in double precision
            //dvec3 oc = dvec3(s.geo.xyz) - r.o;      // Solve t^2*d.d + 2*t*(o-s).d + (o-s).(o-s)-r^2 = 0 
            
            // Solve t^2*d.d + 2*t*(o-s).d + (o-s).(o-s)-r^2 = 0 
            vec3 oc = sub( to_vec3(s.geo), r.o );
            
            //double b=dot(oc,r.d), det=b*b-dot(oc,oc)+s.geo.w*s.geo.w; 
            float b = dot(oc,r.d);
            float det = b*b - dot(oc,oc) + s.geo[3] * s.geo[3]; 
            
            if (det < 0.0f) {
                continue; 
            } else {
                det=sycl::sqrt(det); 
                // det=sqrt(det); 
            }
            d = ( d = ( b - det ) ) > eps ? d : ( ( d = ( b + det ) ) > eps ? d : inf );
            if(d < t) { t=d; id = i; } 
        } 
        if (t < inf) {
            x = add( r.o, mul( r.d, t ) );
            n = normalize(sub(x,to_vec3(pSpheres[id].geo)));
            return true;
        }
        return false;
    }

    // http://www.jcgt.org/published/0009/03/02/
    // https://www.shadertoy.com/view/XlGcRh
    vec3 rand01( uint3& v ) {
        v[0] = v[0] * 1664525u + 1013904223u;
        v[1] = v[1] * 1664525u + 1013904223u;
        v[2] = v[2] * 1664525u + 1013904223u;

        v[0] += v[1]*v[2];
        v[1] += v[2]*v[0];
        v[2] += v[0]*v[1];

        v[0] ^= ( v[0] >> 16u );
        v[1] ^= ( v[1] >> 16u );
        v[2] ^= ( v[2] >> 16u );

        v[0] += v[1]*v[2];
        v[1] += v[2]*v[0];
        v[2] += v[0]*v[1];

        vec3 fval{ static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]) };
        return mul( fval, (1.0f/static_cast<float>(0xffffffffU)) );
    }    

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
    int32_t spp = argc>1 ? atoi(argv[1]) : 100;    // samples per pixel 
    int32_t resy = argc>2 ? atoi(argv[2]) : 500;    // vertical pixel resolution
    int32_t resx = resy; // horiziontal pixel resolution
    
    float fresx = static_cast< float >( resx );
    float fresy = static_cast< float >( resy );
    float recip_spp = 1.0f / static_cast<float>( spp );
    
    auto tstart = std::chrono::system_clock::now();		// take start time

    // The default device selector will select the most performant device.
    auto d_selector{sycl::default_selector_v};

    const auto numSpheres = _countof(spheres);
    std::cout << "numSpheres = " << numSpheres << std::endl;
    
    try {
        sycl::queue q(d_selector, exception_handler);

        // Print out the device information used for the kernel code.
        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // Create arrays with "array_size" to store input and output data. Allocate
        // unified shared memory so that both CPU and device can access them.
        const uint32_t numPixels = resx * resy;
        vec4*   pRadiances = sycl::malloc_shared<vec4>(numPixels, q);
        Sphere* pSpheres   = sycl::malloc_shared<Sphere>(_countof(spheres), q);

        if ( pRadiances == nullptr || pSpheres == nullptr ) {
            std::cout << "Shared memory allocation failure.\n";
            return -1;
        }

        // Initialize radiances with 0.0f
        for (size_t i = 0; i < numPixels; i++) { pRadiances[i] = vec4{ 0.0f, 0.0f, 0.0f, 0.0f }; }

        {
            // Create the range object for the arrays.
            sycl::range<1> num_items{numPixels};

            // Use parallel_for to run vector addition in parallel on device. This
            // executes the kernel.
            //    1st parameter is the number of work items.
            //    2nd parameter is the kernel, a lambda that specifies what to do per
            //    work item. the parameter of the lambda is the work item id.
            // SYCL supports unnamed lambda kernel by default.
            for ( uint32_t pass = 0u; pass < spp; pass++ ) {
                auto e = q.parallel_for(
                    numPixels, 
                    [=](auto gid) { 
                        
                        // determine what pixel we are calculating in this thread
                        // threadId is unique and is [0,pixelCount]
                        uint32_t threadId = static_cast<uint32_t>(gid);
                        uint32_t x = threadId % resx;
                        uint32_t y = threadId / resx;
                        uint32_t addr = x + ( resy - 1 - y ) * resx;
                        
                        //-- define camera
                        Ray cam = Ray{ vec3{ 0.0f, 0.52f, 7.4f }, normalize( vec3{ 0.0f, -0.06f, -1.0f } ) };
                        vec3 cx = normalize( cross( cam.d, abs( cam.d[1] ) < 0.9f ? vec3{ 0.0f,1.0f,0.0f } : vec3{ 0.0f,0.0f,1.0f } ) );
                        vec3 cy = cross( cx, cam.d );
                        //const vec2 sdim = vec2{0.036f, 0.024f};    // sensor size (36 x 24 mm)
                        const vec2 sdim = vec2{0.03f, 0.03f};

                        //-- sample sensor
                        uint3 randSeed{x,y,pass};
                        vec3 rnd3 = rand01(randSeed);
                        
                        vec2 tent = vec2{
                            ( rnd3[0] < 1.0f ) ? ( sycl::sqrt( rnd3[0] ) - 1.0f ) : ( 1.0f - sycl::sqrt( 2.0f - rnd3[0] ) ), 
                            ( rnd3[1] < 1.0f ) ? ( sycl::sqrt( rnd3[1] ) - 1.0f ) : ( 1.0f - sycl::sqrt( 2.0f - rnd3[1] ) ) };
                        
                        float x_lane = ( ( static_cast<float>(x) + 0.5f + ((pass/2)%2) + tent[0] ) / fresx - 0.5f ) * sdim[0];
                        float y_lane = ( ( static_cast<float>(y) + 0.5f + (pass%2)     + tent[1] ) / fresy - 0.5f ) * sdim[1];
                        vec2 s{ x_lane, y_lane };                        
                        //vec2 s = ((x + 0.5f * (0.5f + add( vec2{(pass/2)%2, pass%2}, tent ) )) / vec2{resx, resy} - 0.5f) * sdim;
                        
                        vec3 spos = add( cam.o, add( mul( cx, s[0] ), mul( cy, s[1] ) ) );
                        vec3 lc   = add( cam.o, mul( cam.d, 0.035f ) );           // sample on 3d sensor plane
                        Ray r = Ray{ lc, normalize( sub( lc, spos ) ) };      // construct ray

                        vec3 accrad=vec3{ 0.0f, 0.0f, 0.0f };
                        vec3 accmat=vec3{ 1.0f, 1.0f, 1.0f };        // initialize accumulated radiance and bxdf

                        //-- loop over ray bounces
                        float emissive = 1.0f;
                        for (uint32_t depth = 0, maxDepth = 64; depth < maxDepth; depth++) { 
                            
                            uint32_t id = 0;
                            vec3 isectPos, isectNorm;
                            
                            if (intersect(r, spheres, id, isectPos, isectNorm)) {   // intersect ray with scene
                                const Sphere& obj = spheres[id];
                                
                                vec3 nl = dot( isectNorm, r.d ) < 0.0f ? isectNorm : mul( isectNorm, -1.0f );
                                accrad = add( accrad, mul( accmat, mul( to_vec3( obj.e ), emissive ) ) );      // add emssivie term only if emissive flag is set to 1
                                //accmat *= obj.c.xyz;
                                accmat = mul( accmat, to_vec3( obj.c ) );
                                
                                vec3 rnd = rand01( randSeed );
                                
                                float p = std::max(std::max(obj.c[0], obj.c[1]), obj.c[2]);  // max reflectance
                                
                                if (depth > 5) {
                                    if (rnd[2] >= p) {
                                        break;  // Russian Roulette ray termination
                                    } else {
                                        accmat = mul( accmat, 1.0f / p );       // Energy compensation of surviving rays
                                    }
                                }
                                
                                if (obj.c[3] == 1.0f) { //-- Ideal DIFFUSE reflection
                                    for (int i = 0; i < _countof(spheres); i++) { // Direct Illumination: Next Event Estimation over any present lights
                                        const Sphere& ls = spheres[i];
                                        //if ( all( equal( ls.e, vec3{0.0f, 0.0f, 0.0f } ) ) ) { continue; } // skip non-emissive spheres 
                                        if ( ls.e[0] == 0.0f && ls.e[1] == 0.0f && ls.e[2] == 0.0f ) { continue; } // skip non-emissive spheres 
                                        vec3 xls, nls;
                                        vec3 xc = sub( to_vec3( ls.geo ), isectPos );
                                        vec3 sw = normalize(xc);
                                        vec3 su = normalize(cross((abs(sw[0])>0.1f ? vec3{0.0f,1.0f,0.0f} : vec3{1.0f,0.0f,0.0f}), sw));
                                        vec3 sv = cross(sw,su);
                                        float cos_a_max = sycl::sqrt(float(1.0f - ls.geo[3]*ls.geo[3] / dot(xc,xc)));
                                        float cos_a = 1 - rnd[0] + rnd[0]*cos_a_max;
                                        float sin_a = sycl::sqrt(1.0f - cos_a*cos_a);
                                        float phi = 2.0f * pi * rnd[1];
                                        vec3 l = normalize( add( add( mul( su, cos(phi)*sin_a ), mul( sv, sin(phi)*sin_a ) ), mul( sw, cos_a ) ) );   // sampled direction towards light
                                        uint32_t idls = 0; 
                                        if ( intersect( Ray{ isectPos, l }, spheres, idls, xls, nls ) && idls == i ) {      // test if shadow ray hits this light source
                                            float omega = 2.0f * pi * (1.0f-cos_a_max);
                                            //accrad += accmat / pi * max(dot(l,nl),0.0f) * ls.e * omega;   // brdf term obj.c.xyz already in accmat, 1/pi for brdf
                                            vec3 newAccmat = mul( mul( accmat, 1.0f / pi * std::max(dot(l,nl),0.0f) ), mul( to_vec3( ls.e ), omega ) );
                                            accrad = add( accrad, newAccmat );
                                        }
                                    }
                                    // Indirect Illumination: cosine-weighted importance sampling
                                    const float r1 = 2 * pi * rnd[0];
                                    const float r2 = rnd[1];
                                    const float r2s = sycl::sqrt(r2);
                                    vec3 w = nl;
                                    vec3 u = normalize((cross(abs(w[0])>0.1f ? vec3{0.0f,1.0f,0.0f} : vec3{1.0f,0.0f,0.0f}, w)));
                                    vec3 v = cross(w,u);
                                    r = Ray{ isectPos, normalize( add( mul( u, cos(r1)*r2s ), add( mul( v, sin(r1)*r2s ), mul( w, sycl::sqrt(1.0f - r2) ) ) ) ) };
                                    emissive = 0.0f;   // in the next bounce, consider reflective part only!
                                } else if (obj.c[3] == 2.0f) { //-- Ideal SPECULAR reflection
                                    r = Ray{ isectPos, reflect( r.d, isectNorm ) };  
                                    emissive = 1.0f; 
                                } else if (obj.c[3] == 3.0f) { //-- Ideal dielectric REFRACTION
                                    const bool into = ( isectNorm[0]==nl[0] && isectNorm[1]==nl[1] && isectNorm[2]==nl[2] );
                                    constexpr float nc=1.0f;
                                    constexpr float nt=1.5f;
                                    const float nnt = ( into ? nc/nt : nt/nc );
                                    const float ddn = dot( r.d, nl );
                                    const float cos2t = 1.0f - nnt * nnt * ( 1.0f - ddn * ddn );
                                    if (cos2t >= 0.0f) {     // Fresnel reflection/refraction
                                        const float mulFactor = ( into ? 1.0f : -1.0f ) * ( ddn * nnt + sycl::sqrt( cos2t ) );
                                        const vec3 tDir = normalize( sub( mul( r.d, nnt ), mul( isectNorm, mulFactor ) ) );
                                        const float a = nt - nc;
                                        const float b = nt + nc;
                                        const float R0 = ( a * a ) / ( b * b );
                                        const float c = 1.0f - ( into ? -ddn : dot( tDir, isectNorm ) );
                                        const float Re = R0 + ( 1.0f - R0 ) * c * c * c * c * c;
                                        const float Tr = 1.0f - Re;
                                        const float P = 0.25f + 0.5f*Re;
                                        const float RP = Re / P;
                                        const float TP = Tr / ( 1.0f - P );
                                        r = Ray{ isectPos, rnd[0] < P ? reflect( r.d, isectNorm ) : tDir };      // pick reflection with probability P
                                        //accmat *=  rnd[0] < P ? RP : TP;                     // energy compensation
                                        accmat = mul( accmat, rnd[0] < P ? RP : TP );                                        
                                    } else {
                                        r = Ray{ isectPos, reflect( r.d, isectNorm ) };                      // Total internal reflection
                                    }
                                    emissive = 1.0f; 
                                }
                            
                            } 
                        }

                        // accRad[gid] += vec4(accrad / samps.y, 0);   // <<< accumulate radiance   vvv write 8bit rgb gamma encoded color
                        vec3 scaledRad3 = mul( accrad, recip_spp );
                        pRadiances[ addr ] = add( pRadiances[ addr ], to_vec4( scaledRad3 ) );
                    }
                );

                // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
                // the kernel asynchronously. Wait for the asynchronous call to complete.
                e.wait();            

            }
        }

        auto tend = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count();

        printf( "render duration = %f sec\n", duration );

        //-- write inverse sensor image to file
        FILE *file;
        int err = fileopen(&file, "image-cpu.ppm");
        fprintf(file, "P3\n");
        fprintf(file, "# spp: %d\n", spp);
        fprintf(file, "# rendering time: %f s\n", duration);
        fprintf(file, "%d %d\n%d\n", resx, resy, 255);
        for (int i = resx * resy; i--;) {
            fprintf(file, "%d %d %d ", toInt(pRadiances[i][0]), toInt(pRadiances[i][1]), toInt(pRadiances[i][2]));
        }

        sycl::free(pRadiances, q);
        sycl::free(pSpheres, q);

    } catch (sycl::exception const &e) {
        std::cout << "An exception is caught: " << e.what() << std::endl;
        std::terminate();
    }

}
