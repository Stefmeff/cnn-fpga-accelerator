/**
 * @file conv2d.cpp
 * @author Stefan Moser
 * @brief This file implements a 2D convolution operation using HLS. 
 * 
 * @details The 2D-convolution applies a 3x3 kernel to an input feature map with stride 1 and zero padding.
 * The input feature map is represented as a 3D tensor with dimensions [Cin, H, W], where Cin is the number of input channels, 
 * H is the height, and W is the width. The kernel weights are represented as a 4D tensor with dimensions [Cout, Cin, 3, 3], 
 * where Cout is the number of output channels. The output feature map is also a 3D tensor with dimensions [Cout, H, W].
 * 
 * The goal is to implement multiple versions of the convoltion operation optimized for HLS,
 * 
 */

#include "conv2d.h"

/**
 * TODO:
 * - implement hls for 2d convolution
 * - first implement baseline (simple nexted loops), no optimizations
 * - then HLS optimized version (dataflow, pipeline, array_partition, etc.)
 * - then maybe algorithmic approach (Winograd, FFT, etc.)
 */



/**
 * @brief Baseline implementation of 3x3 convolution (stride 1, zero pad).
 * Pure convolution — no activation. Exposed as the reusable conv core.
 */
void conv2d_core(
        const act_t x[],
        const weight_t w[],
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W)
{
    // Loop over output channels, height, and width
    for (int oc = 0; oc < Cout; oc++) {
        for (int oy = 0; oy < H; oy++) {
            for (int ox = 0; ox < W; ox++) {

                // add bias to acc (wide accumulator type avoids overflow)
                acc_t acc = b[oc];

                // Reduction over input channels. Pipeline this loop (the innermost
                // VARIABLE-bound loop) and fully unroll the constant 3x3 kernel
                // taps below it into a parallel multiply/add tree.
                for (int ic = 0; ic < Cin; ic++) {
                    // No hard II=1: let HLS register the 9-MAC adder tree across
                    // pipeline stages to meet the clock (II=1 forced the whole
                    // tree into one 13.5ns cycle and violated timing).
                    #pragma HLS PIPELINE
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=64

                    for (int ky = 0; ky < KSIZE; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < KSIZE; kx++) {
                            #pragma HLS UNROLL

                            int iy = oy + ky - 1;
                            int ix = ox + kx - 1;

                            // zero-pad borders with a mask instead of `continue`
                            // (control-flow breaks prevent unrolling/pipelining)
                            bool valid = (iy >= 0 && iy < H && ix >= 0 && ix < W);
                            act_t    xv = valid ? x[(ic * H + iy) * W + ix] : (act_t)0;
                            weight_t wv = w[((oc * Cin + ic) * KSIZE + ky) * KSIZE + kx];
                            acc += xv * wv; // accumulate
                        }
                    }
                }
                // write output (requantize accumulator back to activation type)
                z[(oc * H + oy) * W + ox] = (act_t)acc;
            }
        }
    }
}


void conv2d_hls(
        const act_t x[],
        const weight_t w[],
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W
    )
{
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=CONV_MAX_X
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem1 depth=CONV_MAX_W
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2 depth=CONV_MAX_COUT
#pragma HLS INTERFACE m_axi port=z offset=slave bundle=gmem3 depth=CONV_MAX_Z

#pragma HLS INTERFACE s_axilite port=x    bundle=control
#pragma HLS INTERFACE s_axilite port=w    bundle=control
#pragma HLS INTERFACE s_axilite port=b    bundle=control
#pragma HLS INTERFACE s_axilite port=z    bundle=control
#pragma HLS INTERFACE s_axilite port=Cin  bundle=control
#pragma HLS INTERFACE s_axilite port=Cout bundle=control
#pragma HLS INTERFACE s_axilite port=H    bundle=control
#pragma HLS INTERFACE s_axilite port=W    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    conv2d_core(x, w, b, z, Cin, Cout, H, W);
}
