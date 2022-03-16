COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	Include_Path = -I ../
	Flags = -O2 -fopenmp -ftree-vectorize -std=c++14
	Libraries = -O2 -fopenmp
	ArchSuffix=_mc
endif

ifeq ($(COMPILER),nvcc)
	CUDA_DIR = /usr/local/cuda/
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
