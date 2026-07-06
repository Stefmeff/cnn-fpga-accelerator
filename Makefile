OBJDIR=build
SRCDIR=src
UTILDIR=utils
HLSDIR=hls
CONVDIR=hls/conv2d

# Fall back to the default install path if XILINX_VITIS isn't set in the shell
# (needed so the CPU build can find ap_fixed.h when USE_FIXED_POINT is on).
XILINX_VITIS ?= C:/AMDDesignTools/2025.2/Vitis
VITIS_INC=$(XILINX_VITIS)/include
XILINX_XRT=/usr/include/xrt/


USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS= $(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

FLAGS= -I$(SRCDIR) -I$(UTILDIR) -I$(HLSDIR) -I$(CONVDIR)

ifdef BOARD
# ===== FPGA host build (run on the Pynq board): `make BOARD=1` =====
# The convEngine kernel lives in the .xclbin, so the HLS sources (hw_cnn.cpp,
# conv2d.cpp) are NOT compiled here — they need ap_fixed/Vitis headers the board
# doesn't have. The host talks to the PL through XRT.
FLAGS += -std=c++17 -I$(XILINX_XRT) -O3 -DBOARD
LD_FLAGS = -lxrt_coreutil -pthread -lbload
else
# ===== CPU build (HLS C-model, default): plain `make` =====
HLS_SRCS= $(wildcard $(HLSDIR)/*.cpp)
OBJS += $(patsubst $(HLSDIR)/%.cpp,$(OBJDIR)/%.o,$(HLS_SRCS))
# conv2d lives in a subdirectory of hls/, add it explicitly
OBJS += $(OBJDIR)/conv2d.o
FLAGS += -I$(VITIS_INC) -O2
# Flags for SW Development
#FLAGS +=  -I$(VITIS_INC) -g -O0 -fsanitize=address
endif


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
