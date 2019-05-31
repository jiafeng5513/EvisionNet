CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11
LDFLAGS = `pkg-config opencv --libs`

CXXSOURCES=$(wildcard opencvKeypointDetector.cpp sift.cpp)
CXXOBJECTS=$(CXXSOURCES:.cpp=.o)
EXECUTABLE=opencvDetector

CXXSOURCES2=$(wildcard opencvKeypointDescriptor.cpp)
CXXOBJECTS2=$(CXXSOURCES2:.cpp=.o)
EXECUTABLE2=opencvDescriptor


all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE)

release: CFLAGS += -O2 -DNDEBUG
release: $(EXECUTABLE) $(EXECUTABLE2)

$(EXECUTABLE): $(CXXSOURCES) $(CXXOBJECTS)
	# make -C src/liblinear-1.94 lib
	$(CC) $(CXXOBJECTS) $(CFLAGS) $(LDFLAGS) -o $@ 

$(EXECUTABLE2): $(CXXSOURCES2) $(CXXOBJECTS2)
	# make -C src/liblinear-1.94 lib
	$(CC) $(CXXOBJECTS2) $(CFLAGS) $(LDFLAGS) -o $@ 

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(EXECUTABLE)  $(EXECUTABLE2) $(CXXOBJECTS) $(CXXOBJECTS2)
