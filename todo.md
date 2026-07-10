
DATAFLOW region for CONV layer state! => write weights directly into fifo and read in conv2d core

Questions about Project:


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








- Read paper

make → builds default (CPU mode)
./inference t → test mode (correctness check)
./inference b 100 → benchmark 100 images
./inference p images/plane.bmp → predict single image


- Implement a single layer and test Implement => testing of single layer unit (e.g conv2d, maxpool) with kernel.cpp reference implementation



- Folder structure:
-hls/
    - top unit and folders for every component
    -conv2d/
        -testbenches and source code, tcl files to automate vitis workflow
    -relu/
    -etc.


- look at different algorithms/ways of implementing (FFT,)

# NOTES:

# Input Sizes

Dimensions of CIFAR-10:
C × H × W = 3 × 32 × 32 = 3,072 elements

1 float = 4 Bytes => 3,072 × 4 B  = 12,288 B  = 12 KB
3,072 × 32 b = 98,304 bits

The device has 140 BRAM36 blocks able to store 1024 words(floats)
3072/1024 = 3 BRAM Blocks to store one input image:



# Per Layer Output sizes



Largest activation 
= 16×32×32 = 16,384 floats = 64 KB = 16 BRAM36 (layers 0–13). 

| Idx | Layer | Output C×H×W | Elements | float KB | BRAM36 |
|----:|-------|-------------|---------:|---------:|-------:|
| 0 | Conv 3→16 | 16×32×32 | 16,384 | 64 | 16 |
| 1 | ReLU (in-place) | 16×32×32 | 16,384 | — | — |
| 2 | Conv 16→16 | 16×32×32 | 16,384 | 64 | 16 |
| 3 | ReLU | 16×32×32 | 16,384 | — | — |
| 4 | Conv 16→16 | 16×32×32 | 16,384 | 64 | 16 |
| 5 | ReLU | 16×32×32 | 16,384 | — | — |
| 6 | Conv 16→16 | 16×32×32 | 16,384 | 64 | 16 |
| 7 | ReLU | 16×32×32 | 16,384 | — | — |
| 8 | Conv 16→16 | 16×32×32 | 16,384 | 64 | 16 |
| 9 | ReLU | 16×32×32 | 16,384 | — | — |
| 10 | Conv 16→16 | 16×32×32 | 16,384 | 64 | 16 |
| 11 | ReLU | 16×32×32 | 16,384 | — | — |
| 12 | Conv 16→16 | 16×32×32 | 16,384 | 64 | 16 |
| 13 | ReLU | 16×32×32 | 16,384 | — | — |
| 14 | MaxPool | 16×16×16 | 4,096 | 16 | 4 |
| 15 | Conv 16→32 | 32×16×16 | 8,192 | 32 | 8 |
| 16 | ReLU | 32×16×16 | 8,192 | — | — |
| 17 | Conv 32→32 | 32×16×16 | 8,192 | 32 | 8 |
| 18 | ReLU | 32×16×16 | 8,192 | — | — |
| 19 | Conv 32→32 | 32×16×16 | 8,192 | 32 | 8 |
| 20 | ReLU | 32×16×16 | 8,192 | — | — |
| 21 | Conv 32→32 | 32×16×16 | 8,192 | 32 | 8 |
| 22 | ReLU | 32×16×16 | 8,192 | — | — |
| 23 | Conv 32→32 | 32×16×16 | 8,192 | 32 | 8 |
| 24 | ReLU | 32×16×16 | 8,192 | — | — |
| 25 | Conv 32→32 | 32×16×16 | 8,192 | 32 | 8 |
| 26 | ReLU | 32×16×16 | 8,192 | — | — |
| 27 | MaxPool | 32×8×8 | 2,048 | 8 | 2 |
| 28 | Conv 32→64 | 64×8×8 | 4,096 | 16 | 4 |
| 29 | ReLU | 64×8×8 | 4,096 | — | — |
| 30 | Conv 64→64 | 64×8×8 | 4,096 | 16 | 4 |
| 31 | ReLU | 64×8×8 | 4,096 | — | — |
| 32 | Conv 64→64 | 64×8×8 | 4,096 | 16 | 4 |
| 33 | ReLU | 64×8×8 | 4,096 | — | — |
| 34 | Conv 64→64 | 64×8×8 | 4,096 | 16 | 4 |
| 35 | ReLU | 64×8×8 | 4,096 | — | — |
| 36 | Conv 64→64 | 64×8×8 | 4,096 | 16 | 4 |
| 37 | ReLU | 64×8×8 | 4,096 | — | — |
| 38 | Conv 64→64 | 64×8×8 | 4,096 | 16 | 4 |
| 39 | ReLU | 64×8×8 | 4,096 | — | — |
| 40 | AvgPool | 64×1×1 | 64 | 0.25 | 1 |
| 41 | Linear →10 | 1×1×10 | 10 | 0.04 | 1 |
| 42 | Softmax | 1×1×10 | 10 | — | — |




┌───────────────────────── Zynq-7020 (PYNQ-Z2) ─────────────────────────┐
│                                                                        │
│   PROCESSING SYSTEM (PS)                    PROGRAMMABLE LOGIC (PL)     │
│  ┌─────────────────────┐                  ┌────────────────────────┐   │
│  │  ARM Cortex-A9      │   s_axilite      │   convEngine kernel     │   │
│  │  • genSeqConvWeights│◄────────────────►│   control regs + FSM    │   │
│  │    packs W,B → DDR  │  (base addrs,    │   compile-time layer    │   │
│  │  • sets kernel ptrs │   start, done)   │   table (43 entries)    │   │
│  └──────────┬──────────┘                  └───────────┬────────────┘   │
│             │                                         │                │
│  ┌──────────▼──────────┐        AXI-HP (m_axi)        │                │
│  │     DDR3  512 MB     │◄────────────────────────────┘                │
│  │  xin  (gmem0)        │   4 master ports:                            │
│  │  wbuf (gmem1) ~1.07MB│   burst read  xin/wbuf/bbuf                  │
│  │  bbuf (gmem2)        │   burst write zout                           │
│  │  zout (gmem3)        │                                              │
│  └─────────────────────┘                                              │
└────────────────────────────────────────────────────────────────────────┘


                          convEngine  (PL fabric)
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  DDR ══ m_axi ═══►  ┌─────────────┐                                    │
  │  (wbuf)             │   LOAD       │  burst rd, DDR → BRAM             │
  │                     │  weight tile │                                   │
  │                     └──────┬───────┘                                   │
  │                            ▼                                           │
  │                     ┌──────────────┐   weights (BRAM)                  │
  │                     │  W0  ⇄  W1   │   double-buffered per layer/tile  │
  │                     └──────┬───────┘                                   │
  │                            │                                           │
  │  ┌──────────────┐   ┌──────▼───────────────────────────┐  ┌────────┐  │
  │  │ ACT buffer A │──►│         CONV ENGINE               │  │ bias   │  │
  │  │ (BRAM,16blk) │   │  line buffers (BRAM)              │◄─│ B(BRAM)│  │
  │  │              │   │   → 3×3 window (FF, registers)    │  └────────┘  │
  │  │ ACT buffer B │◄──│   → MAC array (DSP48)  Σ over Cin │              │
  │  │ (BRAM,16blk) │   │   → + bias → ReLU (fused)         │              │
  │  └──────────────┘   └───────────────────────────────────┘             │
  │      ▲   │           ┌───────────────────────────────┐                │
  │      │   └──────────►│ POOL / AVGPOOL / LINEAR / SMAX │  (share A⇄B)   │
  │      │               └───────────────────────────────┘                │
  │      │                                                                 │
  │      └─ ping-pong: output of layer L in B becomes input of L+1 (swap)  │
  │                                                                        │
  │  ┌──────────────┐                                                      │
  │  │    STORE     │  BRAM → DDR   ── final logits ONLY ──► DDR (zout)     │
  │  └──────────────┘                                                      │
  └────────────────────────────────────────────────────────────────────────┘

  Tier mapping:   DDR = all weights + I/O    BRAM = W0/W1, A, B, line bufs, bias
                  FF  = 3×3 window, accumulators    DSP = MAC array

  ACT buffer A (BRAM)  ── raster stream of input pixels ──►
        │
        ▼
  ┌───────────────┐   holds K-1 rows          ┌──────────────┐
  │  LINE BUFFERS │──────────────────────────►│  3×3 WINDOW  │  (FF, partitioned)
  │  (BRAM)       │   reconstruct neighborhood │  9 taps      │
  └───────────────┘                            └──────┬───────┘
                                                      │  ×  weights (W0)
                                                      ▼
                            ┌──────────────────────────────────────┐
     loop over Cin  ───►    │   MAC ARRAY  (DSP48)   Σ x·w          │
     tile over Cout ───►    │   parallel across a Cout tile         │
                            └───────────────────┬──────────────────┘
                                                ▼  acc (FF/DSP)
                                        + bias  →  ReLU  (fused)
                                                ▼
                                   ACT buffer B (BRAM)   [output pixel]


# Image /activation arrival order

Tensor X(Cin, H, W) → data[c][h][w] = flat[(c·H + h)·W + w]. So the linear order is:

channel 0:  row0[col0..col31], row1[col0..col31], … row31[…]   ← full 32×32 plane
channel 1:  row0[…], row1[…], … row31[…]
channel 2:  …

# Weights arrival order

Each conv layer's Wt[oc] is a Tensor(Cin, 3, 3) → data[ic][ky][kx]. The packer (cnn_utils.cpp:230-234) copies, per layer, for each oc: the whole Cin×9 block. So the flat wbuf order is:


[ conv layer 0 ][ conv layer 2 ][ conv layer 4 ] …   (conv layers, network order)
        │
        └─ oc = 0:  ic0[ky0kx0 ky0kx1 ky0kx2 ky1kx0 … ky2kx2]  ic1[9] … ic(Cin-1)[9]
           oc = 1:  …
           …

# Biases arrival order

lay->B is Tensor(1,1,Cout) → data[0][0][oc] contiguous. The packer copies Cout biases per conv layer (cnn_utils.cpp:259):


[ layer0: b(oc0) b(oc1) … b(oc15) ][ layer2: b(oc0) … ] …

Line buffer inside convolution engine to access activations more efficiently => similar to sobel

Window = data_t win[3][3] → #pragma HLS ARRAY_PARTITION complete → 9 flip-flops, all readable the same cycle. This is the thing accessed "9 at once." It's in registers precisely so the 9-wide parallel read is free.
Line buffer = data_t buf[2][W] → holds the K−1 = 2 previous rows → BRAM. But it is never read 9-wide — only the current column's two vertical neighbors (buf[0][j], buf[1][j]) are fetched per cycle.