//
// Created by Abdelmajid ID ALI on 21/12/2021.
//

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <iostream>
#include <CL/cl2.hpp>
#include<ctime>

using namespace cl;
using namespace std;


double vect_add_opencl(int DIM, const int *a, const int *b, int *c);

double vect_add_cpu(int DIM, const int *a, const int *b, int *c);

double get_event_time(const Event &event) {
    return ((double) event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
            (double) event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0;
}

#define EVENT_TIME(X) printf("Event  %s: take %g ms.\n", (#X), get_event_time(X))

int main(int argc, char *argv[]) {

    std::string mode = "cpu";
    int DIM = 1000000; // 1M
    if (argc == 2) {
        mode = argv[1];
    } else if (argc > 2) {
        mode = argv[1];
        DIM = atoi(argv[2]);
    }


    // Create variables
    int *a = new int[DIM];
    int *b = new int[DIM];
    int *c = new int[DIM];

    for (int i = 0; i < DIM; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }
    double time_taken;
    if (mode == "gpu")
        time_taken = vect_add_opencl(DIM, a, b, c);

    else if (mode == "cpu")
        time_taken = vect_add_cpu(DIM, a, b, c);

    cout << "Time taken to process " << DIM << " elements  is : " << time_taken << " ms" << endl;

    int errors = 0;
    for (int i = 0; i < DIM; ++i) {
        if (c[i] != (a[i] + b[i])) {
            errors++;
        }
    }
    cout << "Number of errors : " << errors << endl;
    delete[]a;
    delete[]b;
    delete[]c;

    return 0;
}

double vect_add_cpu(int DIM, const int *a, const int *b, int *c) {
    clock_t start = clock();
    for (int i = 0; i < DIM; ++i) {
        c[i] = a[i] + b[i];
    }
    return ((double) (clock() - (start)) / CLOCKS_PER_SEC) * 1000.0;
}

double vect_add_opencl(int DIM, const int *a, const int *b, int *c) {
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

    CommandQueue queue(context, QueueProperties::Profiling);

    // copy A
    Event copyA;
    queue.enqueueWriteBuffer(
            aBuffer, CL_TRUE,
            0, sizeof(int) * DIM,
            a,
            nullptr, &copyA
    );

    // copy B
    Event copyB;
    queue.enqueueWriteBuffer(
            bBuffer, CL_TRUE,
            0, sizeof(int) * DIM,
            b, nullptr, &copyB
    );

    // execution of kernel
    Event executeKernel;
    queue.enqueueNDRangeKernel(
            kernel,
            NullRange,
            NDRange(DIM),
            NullRange,
            nullptr, &executeKernel
    );

    // read results
    Event readC;
    queue.enqueueReadBuffer(
            cBuffer, CL_TRUE, 0,
            sizeof(int) * DIM, c,
            nullptr, &readC
    );

    queue.finish();
    queue.flush();
    auto totalTime = get_event_time(copyA) +
                     get_event_time(copyB) +
                     get_event_time(executeKernel) +
                     get_event_time(readC);

    EVENT_TIME(copyA);
    EVENT_TIME(copyB);
    EVENT_TIME(readC);
    EVENT_TIME(executeKernel);

    return totalTime;
}
