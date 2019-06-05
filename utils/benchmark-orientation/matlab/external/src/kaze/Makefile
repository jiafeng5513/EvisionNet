CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11
LDFLAGS = `pkg-config opencv --libs`

CFLAGS += -Isrc/src/lib

CXXSOURCES=$(wildcard src/src/lib/*.cpp)

CXXSOURCES1=kazedetector.cpp $(CXXSOURCES)
CXXOBJECTS1=$(CXXSOURCES1:.cpp=.o)
EXECUTABLE1=originalKAZEdetector

CXXSOURCES2=kazedescriptor.cpp $(CXXSOURCES)
CXXOBJECTS2=$(CXXSOURCES2:.cpp=.o)
EXECUTABLE2=originalKAZEdescriptor

all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE1) $(EXECUTABLE2)

release: CFLAGS += -O2 -DNDEBUG
release: $(EXECUTABLE1) $(EXECUTABLE2)

$(EXECUTABLE1): $(CXXSOURCES1) $(CXXOBJECTS1)
	$(CC) $(CXXOBJECTS1) $(CFLAGS) $(LDFLAGS) -o $@ 

$(EXECUTABLE2): $(CXXSOURCES2) $(CXXOBJECTS2)
	$(CC) $(CXXOBJECTS2) $(CFLAGS) $(LDFLAGS) -o $@ 

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(COBJECTS) $(CXXOBJECTS1) $(EXECUTABLE1) $(CXXOBJECTS2) $(EXECUTABLE2)
