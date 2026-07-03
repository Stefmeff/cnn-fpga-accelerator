OBJDIR=build
SRCDIR=src
UTILDIR=utils
HLSDIR=hls
CONVDIR=hls/conv2d

VITIS_INC=$(XILINX_VITIS)/include
XILINX_XRT=/usr/include/xrt/


USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS= $(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

HLS_SRCS= $(wildcard $(HLSDIR)/*.cpp)
OBJS += $(patsubst $(HLSDIR)/%.cpp,$(OBJDIR)/%.o,$(HLS_SRCS))

# conv2d lives in a subdirectory of hls/, add it explicitly
OBJS += $(OBJDIR)/conv2d.o

FLAGS= -I$(SRCDIR) -I$(UTILDIR) -I$(HLSDIR) -I$(CONVDIR)

# Flags for SW Development
#FLAGS +=  -I$(VITIS_INC) -g -O0 -fsanitize=address 
FLAGS += -I$(VITIS_INC) -g -O0
# Flags for the board
#FLAGS += -std=c++17 -I${XILINX_XRT} -O3
#LD_FLAGS = -lxrt_coreutil -pthread -lbload


TRG=inference


$(TRG): $(OBJS) 
	$(CXX) $(FLAGS)  $^ -o $@  $(LD_FLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	$(CXX) $(FLAGS) -c $< -o $@ 

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp 
	$(CXX) $(FLAGS) -c $< -o $@ 

$(OBJDIR)/%.o : $(HLSDIR)/%.cpp $(HLSDIR)/%.h
	$(CXX) $(FLAGS) -c $< -o $@

$(OBJDIR)/conv2d.o : $(CONVDIR)/conv2d.cpp $(CONVDIR)/conv2d.h
	$(CXX) $(FLAGS) -c $< -o $@


$(OBJDIR)/%.o : $(UTILDIR)/%.cpp $(UTILDIR)/%.h
	$(CXX) $(FLAGS) -c $< -o $@ 

$(OBJDIR)/%.o : $(UTILDIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@ 


clean:
	$(RM) build/*
	$(RM) $(TRG)
