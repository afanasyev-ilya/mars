compilation:

1) git clone https://github.com/afanasyev-ilya/mars.git

2) cd mars

3) make CXX=nvcc

For non-default CUDA Toolkit install location, please specify CUDA_DIR variable in the makefile.

run:

1) ./bin/mars_cuda_cu -dim 20 -mtx test_mat.csv 

<i>to run on matrix with(or without) h vector provided. In the case when h vector not provided, it is set to all zeroes.</i>

2) ./bin/mars_cuda_cu -dim 100

<i>to run on a larger random matrix (100x100 in this example)</i>

3) ./bin/mars_cuda_cu -help

<i>to get information about all available parameters</i>

4) ./bin/mars_cuda_cu -dim 20 -mtx ./cmake-build-debug/test_mat.csv -batch ./cmake-build-debug/batch.txt -nstreams 5 -ngpu 2

<i> to do a run on multiple input parameters {tmin, tmax, cstep, alpha} using the same matrix and h vector.<br>
Batch file has to be provided, example:<br></i>
0 10 0.001 0.3<br>
0 10 0.001 0.4<br>
0 5 0.001 0.8<br>
10 15 0.001 0.2<br>
10 14 0.001 0.5<br>
20 24 0.001 0.5<br>
0 2 0.0001 0.2<br>
0 12 0.001 0.3<br>
0 1 0.001 0.4<br>
4 8 0.001 0.8<br>
10 12 0.001 0.3<br>
1 7 0.001 0.5<br>
1 2 0.001 0.5<br>
0 2 0.0001 0.3<br>