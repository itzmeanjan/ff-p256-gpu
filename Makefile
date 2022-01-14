CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
INCLUDES = -I./include
PROG = run

# expects user to set DO_RUN variable to either of
# {test, benchmark}
#
# If `DO_RUN=test make` is invoked, compiled binary will
# run test cases only
#
# On other hand, if `DO_RUN=benchmark make` is invoked,
# compiled binary will only run benchmark suite
#
# if nothing is set, none is used, which results into
# compiling binary which neither runs test cases nor runs
# benchmark suite !
DFLAGS = -D$(shell echo $(or $(DO_RUN),nothing) | tr a-z A-Z)

all: $(PROG)

# link
$(PROG): main.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

# compile
main.o: main.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -c $< -o $@

aot_cpu:
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=avx512" main.cpp; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=avx2" main.cpp; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=avx" main.cpp; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=sse4.2" main.cpp; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) $(INCLUDES) -fsycl-targets=spir64_gen -Xs "-device 0x4905" main.cpp

clean:
	find . -name '*.o' -o -name 'a.out' -o -name 'run' -o -name '*.gch' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i -style=Mozilla

cuda:
	# make sure you've built `clang++` with CUDA support
	# check https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda
	clang++ $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DFLAGS) $(INCLUDES) -c main.cpp -o main.o
	clang++ $(SYCLFLAGS) $(SYCLCUDAFLAGS) main.o -o $(PROG)
