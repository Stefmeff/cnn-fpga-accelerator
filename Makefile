OBJDIR=build
SRCDIR=src
UTILDIR=utils

VITIS_INC=$(XILINX_VITIS)/include
XILINX_XRT=/usr/include/xrt/


USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS= $(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

HLS_SRCS= $(wildcard $(HLSDIR)/*.cpp)
OBJS += $(patsubst $(HLSDIR)/%.cpp,$(OBJDIR)/%.o,$(HLS_SRCS))

FLAGS= -I$(SRCDIR) -I$(UTILDIR)

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


$(OBJDIR)/%.o : $(UTILDIR)/%.cpp $(UTILDIR)/%.h
	$(CXX) $(FLAGS) -c $< -o $@ 

$(OBJDIR)/%.o : $(UTILDIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@ 


clean:
	$(RM) build/*
	$(RM) $(TRG)
