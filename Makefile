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

# PRJ_ROOT locates data/ (weights) at runtime; "./" => run from the project dir.
FLAGS= -I$(SRCDIR) -I$(UTILDIR) -I$(HLSDIR) -I$(CONVDIR) -DPRJ_ROOT='"./"'

# Auto-generate header dependencies (.d files) so changing ANY included header
# (e.g. dtypes.h) forces the dependent .o to rebuild. Applied only when compiling.
DEPFLAGS= -MMD -MP

USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS= $(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

ifdef BOARD
# ===== FPGA host build (run on the Pynq board): `make BOARD=1` =====
FLAGS += -std=c++17 -I$(XILINX_XRT) -O3 -DBOARD
LD_FLAGS = -lxrt_coreutil -pthread -lbload
else
# ===== CPU build (HLS C-model, default): plain `make` =====
HLS_SRCS= $(wildcard $(HLSDIR)/*.cpp)
OBJS += $(patsubst $(HLSDIR)/%.cpp,$(OBJDIR)/%.o,$(HLS_SRCS))
# conv2d lives in a subdirectory of hls/, add it explicitly
OBJS += $(OBJDIR)/conv2d.o
FLAGS += -I$(VITIS_INC) -O2
endif


TRG=inference


$(TRG): $(OBJS)
	$(CXX) $(FLAGS)  $^ -o $@  $(LD_FLAGS)

# Object rules: '| $(OBJDIR)' is an order-only prereq that creates build/ first.
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(FLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(HLSDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(FLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)/conv2d.o : $(CONVDIR)/conv2d.cpp | $(OBJDIR)
	$(CXX) $(FLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(UTILDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(FLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Pull in the auto-generated header dependencies (silently ignored on first build).
-include $(OBJS:.o=.d)


clean:
	$(RM) build/*
	$(RM) $(TRG)
