CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11 -Wno-deprecated-register
LDFLAGS = `pkg-config opencv --libs`

CFLAGS += -IoriginalSource/include/  -IoriginalSource/src/

CXXSOURCES1=$(wildcard  mainDesc.cpp originalSource/src/freak.cpp)
CXXOBJECTS1=$(CXXSOURCES1:.cpp=.o)

EXECUTABLE1=originalFREAKdescriptor



all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE)

release: CFLAGS += -O2 -DNDEBUG
release: $(EXECUTABLE1)

$(EXECUTABLE1): $(CXXOBJECTS1)
	$(CC) $(CXXOBJECTS1) $(CFLAGS) $(LDFLAGS) -o $@ 


.cc.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(CCOBJECTS) $(CXXOBJECTS1) $(EXECUTABLE1)
