CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11
LDFLAGS = `pkg-config opencv --libs`

CXXSOURCES=$(wildcard opencvFixedScaleKeypointDescriptor.cpp)
CXXOBJECTS=$(CXXSOURCES:.cpp=.o)
EXECUTABLE=opencvFixedScaleDescriptor


all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE)

release: CFLAGS += -O2 -DNDEBUG
release: $(EXECUTABLE)

$(EXECUTABLE): $(CXXSOURCES) $(CXXOBJECTS)
	# make -C src/liblinear-1.94 lib
	$(CC) $(CXXOBJECTS) $(CFLAGS) $(LDFLAGS) -o $@ 

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(CXXOBJECTS) $(EXECUTABLE)
