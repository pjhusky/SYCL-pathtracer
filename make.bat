@REM we must set the LIB dir, otherwise we get 'LINK : fatal error LNK1104: cannot open file 'msvcrt.lib'
@REM LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\lib\x64;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\lib\um\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.22000.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\\lib\10.0.22000.0\\um\x64;C:\DPCppSYCL\dpcpp_compiler\lib

@REM 
clang++ -fsycl -O3 sycl-pathtracer.cpp -o sycl-pathtracer.exe
@REM clang++ -fsycl sycl-pathtracer.cpp -o sycl-pathtracer.exe

