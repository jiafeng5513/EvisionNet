CC=g++
CFLAGS = `pkg-config opencv fftw3 libconfig --cflags` -std=c++11
LDFLAGS = `pkg-config opencv fftw3 libconfig --libs`

CFLAGS += -Isrc/include


CXXSOURCES1=daisydescriptor.cpp $(filter-out src/src/main.cpp, $(wildcard src/src/*.cpp)) 
CXXOBJECTS1=$(CXXSOURCES1:.cpp=.o)
EXECUTABLE1=originalDAISYdescriptor

all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE1) 

release: CFLAGS += -O2 -DNDEBUG -Wno-write-strings
release: $(EXECUTABLE1)

$(EXECUTABLE1): $(CXXSOURCES1) $(CXXOBJECTS1)
	$(CC) $(CXXOBJECTS1) $(CFLAGS) $(LDFLAGS) -o $@ 


.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(CXXOBJECTS1) $(EXECUTABLE1)
