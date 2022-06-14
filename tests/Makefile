
MPI_CFLAGS:= -Wl,-Bsymbolic-functions -Wl,-z,relro -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpich -g
LIBS:=

HIPSYCL:=syclcc --hipsycl-targets=omp -DGPU_SELECTOR=0
DPCPP:=dpcpp -DGPU_SELECTOR=0

# HIPSYCL_CUDA:=syclcc -DRESTRICT=__restrict --hipsycl-platform=cuda --hipsycl-gpu-arch=sm_60 --cuda-path=$(CUDA_HOME) -DGPU_SELECTOR=1
# HIPSYCL_CUDA:=syclcc -DRESTRICT=__restrict --hipsycl-targets=cuda-nvcxx --hipsycl-nvcxx=/opt/spack/linux-ubuntu22.04-cascadelake/gcc-11.2.0/nvhpc-22.3-x2yra3dinsnxmaaviwi6ftbpaukbzpen/Linux_x86_64/22.3/compilers/bin/nvc++ -DGPU_SELECTOR=1
HIPSYCL_CUDA:=syclcc -DRESTRICT=__restrict --hipsycl-targets=cuda-nvcxx -DGPU_SELECTOR=1

#SYCL:=$(HIPSYCL)
SYCL:=$(HIPSYCL_CUDA)
CFLAGS:=-I../include $(MPI_CFLAGS)


# Avoid making main.o so we can link the test objects against *.o in this
# directory.
TEST_SRCS:=$(wildcard *.cpp)
TEST_OBJS:=$(TEST_SRCS:.cpp=.o)

HEADERS:=$(wildcard ../include/*.h) $(wildcard ./*.h)

.PRECIOUS: %.o

test: test_runner
	./test_runner

test_runner: $(TEST_OBJS)
	$(SYCL) -o $@ $(TEST_OBJS)  $(CFLAGS)  $(LIBS)

%.o: %.cpp %.h $(HEADERS)
	$(SYCL) -c $(CFLAGS) -o $@ $<

%.o: %.cpp $(HEADERS)
	$(SYCL) -c $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm *.o test_runner
