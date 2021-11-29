CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
SYCLAOTFLAGS = -fsycl-default-sub-group-size 32
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

clean:
	find . -name '*.o' -o -name 'a.out' -o -name 'run' -o -name '*.gch' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
