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


static void conv2d_weight_stationary(
        const act_t x[], 
        const weight_t w[], 
        const bias_t b[], 
        act_t z[],
        int Cin, 
        int Cout, 
        int H, 
        int W,
        bool relu)
{
    
    acc_t oacc[COUT_TILE][CONV_MAX_PIX];
    #pragma HLS ARRAY_PARTITION variable=oacc complete dim=1

    //loop over output channels in tiles of COUT_TILE
    for (int oc0 = 0; oc0 < Cout; oc0 += COUT_TILE) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_COUT/COUT_TILE

        for (int ic = 0; ic < Cin; ic++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_CIN

            //store input channel's 3x3 kernel for each of the T output channels in registers
            weight_t kernel[COUT_TILE][KSIZE][KSIZE];
            #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

            //read the T kernels for this input channel into registers (reused across whole input map)
            for (int t = 0; t < COUT_TILE; t++)
                for (int ky = 0; ky < KSIZE; ky++)
                    for (int kx = 0; kx < KSIZE; kx++)
                        #pragma HLS UNROLL
                        kernel[t][ky][kx] = w[(((oc0 + t) * Cin + ic) * KSIZE + ky) * KSIZE + kx];


            //store 3 rows of input channel in a line buffer (shift register) for acces accross T tiles of output channels.
            act_t row[KSIZE][CONV_MAX_DIM];
            #pragma HLS ARRAY_PARTITION variable=row complete dim=0

            //initialize the line buffer with pad + first two rows of input
            for (int c = 0; c < W; c++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=8 max=CONV_MAX_DIM
                row[0][c] = (act_t)0;                                       // row -1 (pad)
                row[1][c] =            x[(ic * H + 0) * W + c];             // row 0
                row[2][c] = (1 < H) ? x[(ic * H + 1) * W + c] : (act_t)0;   // row 1
            }

            //slide the 3x3 window across the input feature map
            for (int oy = 0; oy < H; oy++) {
                #pragma HLS LOOP_TRIPCOUNT min=8 max=CONV_MAX_DIM

                //input row streamed in during this pass, feeding the next output row
                int next = oy + 2;

                //slide the 3x3 window across the row
                for (int ox = 0; ox < W; ox++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=8 max=32

                    //comput accross T output channels in parallel
                    for (int t = 0; t < COUT_TILE; t++) {
                        #pragma HLS UNROLL
                        acc_t psum = 0;

                        //parallel multiply-accumulate for this unit's 3x3 window
                        for (int ky = 0; ky < KSIZE; ky++) {
                            #pragma HLS UNROLL
                            for (int kx = 0; kx < KSIZE; kx++) {
                                #pragma HLS UNROLL
                                int ix = ox + kx - 1;
                                act_t xv = (ix >= 0 && ix < W) ? row[ky][ix] : (act_t)0;
                                psum += xv * kernel[t][ky][kx];
                            }
                        }

                        //either add bias for the first input channel or accumulate with previous input channels
                        acc_t prev = (ic == 0) ? (acc_t)b[oc0 + t] : oacc[t][oy * W + ox];
                        oacc[t][oy * W + ox] = prev + psum;
                    }


                    if (ox >= 1) {
                        int c = ox - 1;
                        row[0][c] = row[1][c];
                        row[1][c] = row[2][c];
                        row[2][c] = (next < H) ? x[(ic * H + next) * W + c] : (act_t)0;
                    }
                }

                {
                    int c = W - 1;
                    row[0][c] = row[1][c];
                    row[1][c] = row[2][c];
                    row[2][c] = (next < H) ? x[(ic * H + next) * W + c] : (act_t)0;
                }
            }
        }

        //write accumulated results to output (include relu)
        for (int t = 0; t < COUT_TILE; t++) {
            for (int i = 0; i < H * W; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=1024
                acc_t v = oacc[t][i];
                if (relu && v < (acc_t)0) v = (acc_t)0;   // fused ReLU
                z[(oc0 + t) * H * W + i] = (act_t)v;
            }
        }
    }
}

/** Dispatch to the dataflow variant selected by CONV_DATAFLOW (see conv2d.h). */
void conv2d_core(
        const act_t x[],
        const weight_t w[],
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W,
        bool relu)
{
    //Change version in here:
    conv2d_weight_stationary(x, w, b, z, Cin, Cout, H, W, relu);
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
