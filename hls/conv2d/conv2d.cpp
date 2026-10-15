#include "conv2d.h"
#include <hls_stream.h>

/**
 * @brief Performs a convolution layer (+RELU) with the given parameters.
 * 
 * @param x buffer storing the input feature map
 * @param w fifo stream of weights
 * @param b bias buffer
 * @param z output buffer
 * @param Cin number of input channels
 * @param Cout number of output channels
 * @param H height of the input feature map
 * @param W width of the input feature map
 */
void conv2d_ws(
        const act_t x[],
        hls::stream<wtile>& w,
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W)
{
    //BRAM: stores the accumulated output values for the current tile of output channels
    acc_t oacc[COUT_TILE][CONV_MAX_PIX];
    #pragma HLS ARRAY_PARTITION variable=oacc complete dim=1
    #pragma HLS bind_storage variable=oacc type=ram_2p impl=bram

    int w_read = 0;
    //iterate over output channels in tiles => process COUT_TILE output channels at a time
    for (int oc0 = 0; oc0 < Cout; oc0 += COUT_TILE) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_COUT/COUT_TILE

    
        for (int ic = 0; ic < Cin; ic++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_CIN

            //Register for filter kernel of each output channel of the tile
            weight_t kernel[COUT_TILE][KSIZE][KSIZE];
            #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

            //load weights of from fifo
            //each fifo read returns weight at (ky,kx) for all output channels of the tile (COUT_TILE)
            init_kernel_y: for (int ky = 0; ky < KSIZE; ky++) {
                init_kernel_x: for (int kx = 0; kx < KSIZE; kx++) {
                    #pragma HLS PIPELINE II=1
                    wtile col = w.read();
                    for (int t = 0; t < COUT_TILE; t++) {
                        #pragma HLS UNROLL
                        kernel[t][ky][kx] = col.w[t];
                    }
                }
            }

            //line buffer to store the first two rows of the input feature map of current input channel
            //reused for all output channels of the tile
            act_t line_buf[KSIZE - 1][CONV_MAX_DIM];
            #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

            //initialize: first row => padding, second row => read input fmap
            init_line_buf: for (int c = 0; c < W; c++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=8 max=CONV_MAX_DIM
                line_buf[0][c] = (act_t)0;                       
                line_buf[1][c] = x[ic * H * W + c];              
            }

            //3x3 window to efficiently access fmap values
            act_t window[KSIZE][KSIZE];
            #pragma HLS ARRAY_PARTITION variable=window complete dim=0


            //various indices for iteration:
            int oy = 0, cx = 0;                  
            int out_row = 0;                     
            const int ic_base = ic * H * W;
            int rd_idx = ic_base + 1 * W;        
            const int in_end = ic_base + H * W;

            //sliding filter over whole input fmap
            sliding_window: for (int p = 0; p < H * (W + 1); p++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=oacc inter false
                #pragma HLS LOOP_TRIPCOUNT min=72 max=CONV_MAX_PIX+CONV_MAX_DIM

                //read next pixel from input fmap (or 0 if padding)
                bool reading = (cx < W);
                int  rc = reading ? cx : 0;
                int safe_rd_idx = (rd_idx < in_end) ? rd_idx : (in_end - 1);
                act_t new_pixel = (reading && rd_idx < in_end) ? x[safe_rd_idx] : (act_t)0;

                //read next column form line buffer (or 0 if padding)
                act_t r0 = reading ? line_buf[0][rc] : (act_t)0;
                act_t r1 = reading ? line_buf[1][rc] : (act_t)0;
                act_t r2 = new_pixel;

                //shift window left and insert column of new pixels
                for (int ky = 0; ky < KSIZE; ky++) {
                    #pragma HLS UNROLL
                    window[ky][0] = window[ky][1];
                    window[ky][1] = window[ky][2];
                }

                window[0][2] = (oy - 1 >= 0) ? r0 : (act_t)0;
                window[1][2] = r1;
                window[2][2] = (oy + 1 < H)  ? r2 : (act_t)0;

                if (reading) {
                    line_buf[0][rc] = r1;
                    line_buf[1][rc] = r2;
                }

                //compute output for the current window for all output channels of the tile
                int oc = cx - 1;                 
                if (oc >= 0) {
                    int p_out = out_row + oc;
                    compute: for (int t = 0; t < COUT_TILE; t++) {
                        #pragma HLS UNROLL
                        acc_t psum = 0;
                        //compute partial sum for whole 3x3 window
                        for (int ky = 0; ky < KSIZE; ky++) {
                            #pragma HLS UNROLL
                            for (int kx = 0; kx < KSIZE; kx++) {
                                #pragma HLS UNROLL
                                #pragma HLS BIND_OP variable=psum op=addmul impl=dsp
                                int ix = oc + kx - 1;
                                act_t xv = (ix >= 0 && ix < W) ? window[ky][kx] : (act_t)0;
                                psum += xv * kernel[t][ky][kx];
                            }
                        }

                        //read accumulated values from previous computation and add current psum
                        acc_t prev = (ic == 0) ? (acc_t)b[oc0 + t] : oacc[t][p_out];
                        acc_t res  = prev + psum;
                        oacc[t][p_out] = res;
                    }
                }

                if (reading) rd_idx++;
                if (cx == W) { cx = 0; oy++; out_row += W; }
                else         cx++;
            }
        }

        //write result back to output buffer + add relu
        write_back: for (int t = 0; t < COUT_TILE; t++) {
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
        hls::stream<wtile>& w,
        const bias_t b[],
        act_t z[],
        int Cin,
        int Cout,
        int H,
        int W)
{
    conv2d_ws(x, w, b, z, Cin, Cout, H, W);
}

void conv2d_hls(
        const float x[],
        const float w[],
        const float b[],
        act_t z[],               
        int Cin,
        int Cout,
        int H,
        int W)
{
#pragma HLS INTERFACE m_axi port=x bundle=gmem0 depth=65536
#pragma HLS INTERFACE m_axi port=w bundle=gmem1 depth=36864
#pragma HLS INTERFACE m_axi port=b bundle=gmem2 depth=64
#pragma HLS INTERFACE m_axi port=z bundle=gmem3 depth=65536
#pragma HLS INTERFACE s_axilite port=x    bundle=control
#pragma HLS INTERFACE s_axilite port=w    bundle=control
#pragma HLS INTERFACE s_axilite port=b    bundle=control
#pragma HLS INTERFACE s_axilite port=z    bundle=control
#pragma HLS INTERFACE s_axilite port=Cin  bundle=control
#pragma HLS INTERFACE s_axilite port=Cout bundle=control
#pragma HLS INTERFACE s_axilite port=H    bundle=control
#pragma HLS INTERFACE s_axilite port=W    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static act_t  x_buf[CONV_MAX_X];
    static bias_t b_buf[CONV_MAX_COUT];
    static act_t  z_buf[CONV_MAX_Z];

    const int xn = Cin * H * W;
    const int wn = Cout * Cin * 9;
    const int zn = Cout * H * W;

    for (int i = 0; i < xn; i++)   x_buf[i] = (act_t)x[i];
    for (int i = 0; i < Cout; i++) b_buf[i] = (bias_t)b[i];

    // Pack scalar weights ([oc0][ic][ky][kx][t] order) into COUT_TILE-wide words.
    hls::stream<wtile> w_fifo;
    #pragma HLS STREAM variable=w_fifo depth=4096
    const int wwords = wn / COUT_TILE;
    for (int i = 0; i < wwords; i++) {
        wtile col;
        for (int t = 0; t < COUT_TILE; t++)
            col.w[t] = (weight_t)w[i * COUT_TILE + t];
        w_fifo.write(col);
    }

    //conv2d_core(x_buf, w_fifo, b_buf, z_buf, Cin, Cout, H, W);

    for (int i = 0; i < zn; i++) z[i] = z_buf[i];
}
