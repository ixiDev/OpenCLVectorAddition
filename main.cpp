//
// Created by Abdelmajid ID ALI on 21/12/2021.
//

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <iostream>
#include <CL/cl2.hpp>


using namespace cl;
using namespace std;



// A[i] = B[i] + C[i]

int main() {


    // select platform
    std::vector<Platform> platforms;
    Platform::get(&platforms);
    if (platforms.empty()) {
        cout << "No platform found !" << endl;
        exit(1);
    }

    Platform selectedPlatform = platforms[0];
    cout << "Selected Platform is  : " << selectedPlatform.getInfo<CL_PLATFORM_NAME>() << endl;

    // select OCL device
    std::vector<Device> devices;
    selectedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        cout << "OCL devices empty !" << endl;
        exit(1);
    }

    Device target = devices[0];
    cout << "Selected device : " << target.getInfo<CL_DEVICE_NAME>() << endl;


    // Create variables
    int DIM = 16;
    int *a = new int[DIM];
    int *b = new int[DIM];
    int *c = new int[DIM];

    for (int i = 0; i < DIM; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }


    // create and build kernel
    string kernel_source = "kernel void vadd(\n"
                           "    global int *a,\n"
                           "    global int *b,\n"
                           "    global int *c\n"
                           "){\n"
                           "    int i = get_global_id(0);\n"
                           "    c[i] = a[i] + b[i];\n"
                           "}";

    Context context(devices);
    Program program(context, kernel_source);

    auto build_result = program.build(devices);

    if (build_result != CL_SUCCESS) {
        cout << "Build program error " << endl;

        auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (const auto &item: log) {
            cout << "LOG : " << item.second << endl;
        }

        exit(1);
    }

    // Create buffers and kernel args
    Buffer aBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * DIM);
    Buffer bBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * DIM);
    Buffer cBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * DIM);

    Kernel kernel(program, "vadd");

    kernel.setArg(0, aBuffer);
    kernel.setArg(1, bBuffer);
    kernel.setArg(2, cBuffer);

    CommandQueue queue(context);

    // copy A
    queue.enqueueWriteBuffer(
            aBuffer, CL_TRUE,
            0, sizeof(int) * DIM,
            a
    );

    // copy B
    queue.enqueueWriteBuffer(
            bBuffer, CL_TRUE,
            0, sizeof(int) * DIM,
            b
    );

    // execution of kernel
    queue.enqueueNDRangeKernel(
            kernel,
            NullRange,
            NDRange(DIM),
            NullRange
    );
    // read results
    queue.enqueueReadBuffer(
            cBuffer, CL_TRUE, 0,
            sizeof(int) * DIM, c
    );

    queue.finish();
    queue.flush();


    for (int i = 0; i <DIM; ++i) {
        cout << a[i] << "+" << b[i] << " = " << c[i]<<endl;
    }

    delete []a;
    delete []b;
    delete []c;

    return 0;
}
