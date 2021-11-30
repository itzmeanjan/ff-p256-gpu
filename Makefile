CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
INCLUDES = -I./include
PROG = run

$(PROG): main.o test.o ntt.o utils.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

ntt.o: ntt.cpp include/ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

test.o: test/test.cpp include/test.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

main.o: main.cpp include/test.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c $<

aot_cpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c test/test.cpp -o test.o $(INCLUDES)
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx512" ntt.cpp test.o utils.o main.o; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx2" ntt.cpp test.o utils.o main.o; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx" ntt.cpp test.o utils.o main.o; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=sse4.2" ntt.cpp test.o utils.o main.o; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c test/test.cpp -o test.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device 0x4905" ntt.cpp test.o utils.o main.o

clean:
	find . -name '*.o' -o -name 'a.out' -o -name 'run' -o -name '*.gch' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
