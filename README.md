# OpenCLVectorAddition
Vector Addition using OpenCL  CPP 

Tutorial video https://youtu.be/B-YApTvoxa0

#### How to use

```shell
   cd $HOME && git clone https://github.com/ixiDev/OpenCLVectorAddition.git
   cd ~/OpenCLVectorAddition
   cmake -S . -B build
   cmake --build build --target all
```
To execute the program run this : 
```shell
     ./build/OCLTPTut gpu 1024
```
change **gpu** with **cpu** to execute in cpu and 
the second argument is the vector size

##### Output Example :
Run in CPU-GPU
```text
Selected Platform is  : Intel Gen OCL Driver
Selected device : Intel(R) HD Graphics IvyBridge M GT2
Event  copyA: take 1.25072 ms.
Event  copyB: take 1.62272 ms.
Event  readB: take 0.70952 ms.
Event  executeKernel: take 0.6272 ms.
Time taken to process 1000000 elements  is : 4.21016 ms
Number of errors : 0
```
Run in CPU Only
```text
Time taken to process 1000000 elements  is : 3.713 ms
Number of errors : 0
```
