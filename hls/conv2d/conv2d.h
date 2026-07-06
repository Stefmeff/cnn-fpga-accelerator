#ifndef HLS_CONV2D_H
#define HLS_CONV2D_H

#include "dtypes.h"   // act_t / weight_t / bias_t / acc_t

//dimensions derived from utils/nets.h and cifar10 dataset:

#define CONV_MAX_CIN   64                                     // max input channels 
#define CONV_MAX_COUT  64                                     // max output channels
#define CONV_MAX_DIM   32                                     // max H, W (32x32 input images for cifar10)
#define CONV_MAX_PIX   (CONV_MAX_DIM * CONV_MAX_DIM)          // max pixels per channel (32x32)
#define CONV_MAX_X     (CONV_MAX_CIN  * CONV_MAX_PIX)         // max input feature map size 
#define CONV_MAX_Z     (CONV_MAX_COUT * CONV_MAX_PIX)         // max output feature map size
#define CONV_MAX_W     (CONV_MAX_COUT * CONV_MAX_CIN * 9)     // max amount of weights
#define KSIZE 3


// T output channels computed in parallel from the same input window.
// T must divid the CNNs Cout dims (16/32/64) => 1,2,4,8,16
#ifndef COUT_TILE
  #define COUT_TILE 8
#endif

/**
 * @brief Performs a 3x3 convolution operation on the input feature map, with 
 * stride 1 and 0 padding.
 * 
 * @param x Input feature map [Cin*H*W]
 * @param w Weights [Cout*Cin*9]
 * @param b Bias [Cout]
 * @param z Output feature map [Cout*H*W]
 * @param Cin Number of input channels
 * @param Cout Number of output channels
 * @param H Height of input feature map
 * @param W Width of input feature map
 */
void conv2d_hls(
        const act_t x[],
        const weight_t w[],
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W
);


void conv2d_core(
        const act_t x[],
        const weight_t w[],
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W,
        bool relu = false
);
#endif // HLS_CONV2D_H
