# Project Overview

Implemented CNN completely but no real optimizations yet, only general architecture
need to introduce efficient pipelining and data use

TODO:
- finish reading paper
- look at Winograd or FFT for convolution

Build the .xclbin (host machine, Vitis):

# link the HLS kernel object into a hardware .xclbin using link.cfg
v++ -l -t hw --config config/link.cfg convEngine.xo -o convEngine.xclbin


Board Connection:

ping pynq            # or ping 192.168.2.99

ssh xilinx@192.168.2.99
# password: xilinx

# make a target dir on the board first: 
ssh xilinx@192.168.2.99 "mkdir -p /home/xilinx/cnn"

# copy the built artifacts — host exe + xclbin (adjust names to your build output)
scp .\build\host .\build\convEngine.xclbin xilinx@192.168.2.99:/home/xilinx/cnn/

# ...or copy the whole project folder recursively
scp -r .\project\cnn_fpga_accelerator xilinx@192.168.2.99:/home/xilinx/cnn/

ssh xilinx@192.168.2.99
cd /home/xilinx/cnn

# build/link the host natively on the board (XRT + libbload, HLS sources excluded)
make clean && make BOARD=1

bitLoad -p design.bit      # program the PL
./inference t              # run — test / benchmark / predict mode





