CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11 -Wno-deprecated-register
LDFLAGS = `pkg-config opencv --libs`

CFLAGS += -IoriginalSource/include/ -IoriginalSource/thirdparty/agast/include/ -IoriginalSource/thirdparty/agast/include/agast -march=native

CXXSOURCES1=$(wildcard  mainKp.cpp originalSource/src/brisk.cpp)
CXXOBJECTS1=$(CXXSOURCES1:.cpp=.o)
# CCSOURCES=$(originalSource/thirdparty/agast/src/ %.cc,%.o,$(wildcard *.cc))
CCSOURCES=$(wildcard originalSource/thirdparty/agast/src/agast5_8.cc \
					 originalSource/thirdparty/agast/src/oast9_16.cc \
					 originalSource/thirdparty/agast/src/agast7_12d.cc \
					 originalSource/thirdparty/agast/src/agast5_8_nms.cc \
					 originalSource/thirdparty/agast/src/oast9_16_nms.cc \
					 originalSource/thirdparty/agast/src/agast7_12d_nms.cc)
CCOBJECTS=$(CCSOURCES:.cc=.o)
EXECUTABLE1=originalBRISKdetector

CXXSOURCES2=$(wildcard  mainDesc.cpp originalSource/src/brisk.cpp)
CXXOBJECTS2=$(CXXSOURCES2:.cpp=.o)
EXECUTABLE2=originalBRISKdescriptor



all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE)

release: CFLAGS += -O2 -DNDEBUG
release: $(EXECUTABLE1) $(EXECUTABLE2)

$(EXECUTABLE1): $(CCOBJECTS) $(CXXOBJECTS1)
	$(CC) $(CXXOBJECTS1) $(CCOBJECTS)  $(CFLAGS) $(LDFLAGS) -o $@ 

$(EXECUTABLE2): $(CCOBJECTS) $(CXXOBJECTS2)
	$(CC) $(CXXOBJECTS2) $(CCOBJECTS)  $(CFLAGS) $(LDFLAGS) -o $@ 

.cc.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(CCOBJECTS) $(CXXOBJECTS1) $(EXECUTABLE1) $(CXXOBJECTS2) $(EXECUTABLE2)
