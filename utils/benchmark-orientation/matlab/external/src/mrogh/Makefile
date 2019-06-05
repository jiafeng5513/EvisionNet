CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11
LDFLAGS = `pkg-config opencv --libs`

CFLAGS += -Isrc


CXXSOURCES1=mroghdescriptor.cpp src/mrogh.cpp
CXXOBJECTS1=$(CXXSOURCES1:.cpp=.o)
EXECUTABLE1=originalMROGHdescriptor

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
