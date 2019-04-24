CXX=g++
# CFLAGS will be the options passed to the compiler.
INCLUDE=-I./eigen/ -I./boost/
CXXFLAGS=-c -O3 -fopenmp -std=gnu++11 $(INCLUDE)

all: match

match:main.o matcher.o
	$(CXX) -O3 -fopenmp main.o matcher.o -o match -L./boost/stage/lib -lboost_system -lboost_filesystem

main.o:main.cpp
	$(CXX) $(CXXFLAGS) main.cpp

matcher_int.o:matcher.cpp
	$(CXX) $(CXXFLAGS) matcher.cpp

clean:
	rm match main.o matcher.o
