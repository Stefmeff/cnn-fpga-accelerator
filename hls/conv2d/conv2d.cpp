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
#include <hls_stream.h>



static void conv2d_weight_stationary(
        const act_t x[],
        hls::stream<weight_t>& w,
        const bias_t b[],
        act_t z[],
        int Cin, 
        int Cout, 
        int H, 
        int W)
{
    //initialize LUTRAM for accumulated outputs of COUT_TILE output channels
    acc_t oacc[COUT_TILE][CONV_MAX_PIX];
    #pragma HLS ARRAY_PARTITION variable=oacc complete dim=1
    #pragma HLS BIND_STORAGE variable=oacc type=ram_2p impl=lutram


    for (int oc0 = 0; oc0 < Cout; oc0 += COUT_TILE) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_COUT/COUT_TILE
        for (int ic = 0; ic < Cin; ic++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_CIN

            //initialize Register for 3x3 kernel weights for COUT_TILE output channels
            weight_t kernel[COUT_TILE][KSIZE][KSIZE];
            #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

            const int TOTAL_WEIGHTS = COUT_TILE * KSIZE * KSIZE;
            weight_t* kernel_ptr = &kernel[0][0][0];

            load_weights: for (int i = 0; i < TOTAL_WEIGHTS; i++) {
                #pragma HLS PIPELINE II=1
                kernel_ptr[i] = w.read();
            }


            act_t line_buf[KSIZE - 1][CONV_MAX_DIM];
            #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

            for (int c = 0; c < W; c++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=8 max=CONV_MAX_DIM
                line_buf[0][c] = (act_t)0;
                line_buf[1][c] = (ic * H + 0 < H) ? x[(ic * H + 0) * W + c] : (act_t)0;
            }

            act_t window[KSIZE][KSIZE];
            #pragma HLS ARRAY_PARTITION variable=window complete dim=0

            int oy = 0, ox = 0;
            const int ic_base = ic * H * W;
            int rd_idx = ic_base + 1 * W; 
            const int in_end = ic_base + H * W;

            // Schleife über alle Pixel
            for (int p = 0; p < H * W; p++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=oacc inter false
                #pragma HLS LOOP_TRIPCOUNT min=64 max=CONV_MAX_PIX

                act_t new_pixel = (rd_idx < in_end && ox < W) ? x[rd_idx] : (act_t)0;

                
                act_t r0 = line_buf[0][ox];
                act_t r1 = line_buf[1][ox];
                act_t r2 = new_pixel;

                for (int ky = 0; ky < KSIZE; ky++) {
                    #pragma HLS UNROLL
                    window[ky][0] = window[ky][1];
                    window[ky][1] = window[ky][2];
                }
                
                window[0][2] = (oy - 1 >= 0) ? r0 : (act_t)0;
                window[1][2] = r1;
                window[2][2] = (oy + 1 < H)  ? r2 : (act_t)0;

                for (int t = 0; t < COUT_TILE; t++) {
                    #pragma HLS UNROLL
                    acc_t psum = 0;
                    for (int ky = 0; ky < KSIZE; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < KSIZE; kx++) {
                            #pragma HLS UNROLL
                            int ix = ox + kx - 1;
                            act_t xv = (ix >= 0 && ix < W) ? window[ky][kx] : (act_t)0;
                            
                            psum += xv * kernel[t][ky][kx];
                        }
                    }
                    
                    acc_t prev = (ic == 0) ? (acc_t)b[oc0 + t] : oacc[t][p];
                    oacc[t][p] = prev + psum;
                }

                line_buf[0][ox] = r1;
                line_buf[1][ox] = r2;

                rd_idx++;
                if (ox == W - 1) { 
                    ox = 0; 
                    oy++; 
                } else { 
                    ox++; 
                }
            }

        }

        bool final1 = ((Cin & 1) == 0);
        for (int t = 0; t < COUT_TILE; t++) {
            for (int i = 0; i < H * W; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=1024
                acc_t v = oacc[t][i];
                if (v < (acc_t)0) v = (acc_t)0;
                z[(oc0 + t) * H * W + i] = (act_t)v;
            }
        }
    }
}


void conv2d_core(
        const act_t x[],
        hls::stream<weight_t>& w,
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W)
{
    conv2d_weight_stationary(x, w, b, z, Cin, Cout, H, W);
}
