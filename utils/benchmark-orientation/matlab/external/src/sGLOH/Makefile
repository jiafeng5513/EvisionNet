CC=g++
CFLAGS = `pkg-config opencv fftw3 libconfig --cflags` -std=c++11
LDFLAGS = `pkg-config opencv fftw3 libconfig --libs`

CFLAGS += -Isrc/src

CSOURCES1=src/src/sgloh.c
CXXSOURCES1=sglohdescriptor.cpp
COBJECTS1=$(CSOURCES1:.c=.o)
CXXOBJECTS1=$(CXXSOURCES1:.cpp=.o)
EXECUTABLE1=originalsGLOHdescriptor

all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE1) 

release: CFLAGS += -O2 -DNDEBUG -Wno-write-strings
release: $(EXECUTABLE1)

$(EXECUTABLE1): $(CXXSOURCES1) $(CXXOBJECTS1) $(CSOURCES1) $(COBJECTS1)
	$(CC) $(COBJECTS1) $(CXXOBJECTS1) $(CFLAGS) $(LDFLAGS) -o $@ 


.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(CXXOBJECTS1) $(COBJECTS1) $(EXECUTABLE1)
