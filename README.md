# Project Overview

Current Project Layout Idea: 
- create components for different network layers (e.g conv2d, relu, maxpool, etc) in seperate subfolders in hls/ directory
- in each directory add .cpp, .h, and testbench file for the specific hls component (for testbench test against reference solution in kernels.cpp)
- add component to hls.cfg file for c-sim,c-synthesis (for vitis uncomment the component you want to simulate)
- start with simple baseline implementation to test setup => later add improvements

TODO:
- finish reading paper
- look at Winograd or FFT for convolution
