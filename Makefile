COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	Include_Path = -I ../
	Flags = -D __USE_OMP__ $(MPI_Flags) -O3 -fopenmp -ftree-vectorize -std=c++17 -mtune=a64fx -fopenmp -msve-vector-bits=512 -march=armv8.2-a+sve -Ofast -funroll-loops -ffast-math
	Libraries = -O3 -fopenmp
	ArchSuffix=_mc
endif

ifeq ($(COMPILER),nvcc)
	CUDA_DIR = /opt/cuda/cuda-10.1/
	CUDA_COMPILER = $(CUDA_DIR)/bin/nvcc
	Include_Path = -I $(CUDA_DIR)/include -I ../external_libraries/cub -I ../
	Flags = -O2 -D __USE_CUDA__ -x cu -w -m64 -std=c++11 -Xptxas -dlcm=ca --expt-extended-lambda -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp
	Library_Path = -L $(CUDA_DIR)/lib -L $(CUDA_DIR)/lib64
	Libraries = -lcudart -lcudadevrt -lcudadevrt -Xcompiler -fopenmp -lcurand
	ArchSuffix=_cu
endif

.DEFAULT_GOAL := all

##########
# binaries
##########

all: mars_cuda

mars_cuda: create_folders main.o
	$(CXX) object_files/main.o $(Library_Path) $(Libraries) -o ./bin/mars_cuda$(ArchSuffix)

##########
# CPPs
##########

main.o: main.cpp
	$(CXX) $(Flags) $(Include_Path) -c main.cpp -o object_files/main.o

create_folders:
	-mkdir -p ./bin
	-cp graph_library.sh ./bin
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rm -f bin/*$(ArchSuffix)*
