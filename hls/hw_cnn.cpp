#include "hw_cnn.h"
#include "dtypes.h"
#include "conv2d/conv2d.h"
#include <hls_stream.h>

#define N_LAYERS 24

enum Layer_t { CONV, MAXPOOL, AVGPOOL, LINEAR, SOFTMAX };

struct LayerParams {
    Layer_t type;
    int Cin, Cout, H, W;
};

static const LayerParams CNN_Layers[N_LAYERS] = {
    {CONV, 3, 16, 32, 32},
    {CONV, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32},
    {MAXPOOL, 16, 16, 32, 32},
    {CONV, 16, 32, 16, 16},
    {CONV, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16},
    {MAXPOOL, 32, 32, 16, 16},
    {CONV, 32, 64, 8, 8},
    {CONV, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},
    {AVGPOOL, 64, 64, 8, 8},
    {LINEAR, 64, 10, 1, 1},
    {SOFTMAX, 10, 10, 1, 1},
};


/**
 * @brief Copies n floats from off-chip DRAM (src) into an on-chip BRAM(dst).
 * Two elements per iteration, convert to respective data type T.
 */
template<typename T>
static void load_dram(const float* src, T* dst, int n) {
    int pairs = n >> 1;
    for (int i = 0; i < pairs; i++) {
        #pragma HLS PIPELINE II=2
        dst[2*i]     = (T)src[2*i];
        dst[2*i + 1] = (T)src[2*i + 1];
    }
    if (n & 1)
        dst[n-1] = (T)src[n-1];
}

/**
 * @brief Copies n elements from an on-chip buffer (src) back to DRAM (dst).
 */
template<typename T>
static void store_dram(const T* src, float* dst, int n) {
    int pairs = n >> 1;
    for (int i = 0; i < pairs; i++) {
        #pragma HLS PIPELINE II=2
        dst[2*i]     = (float)src[2*i];
        dst[2*i + 1] = (float)src[2*i + 1];
    }
    if (n & 1)
        dst[n-1] = (float)src[n-1];
}

/**
 * @brief Streams weights from DRAM to a fifo. 
 * Two elements per iteration, convert to weight_t.
 * 
 * @param src off-chip DRAM
 * @param ws FIFO stream for weights
 * @param n number of weights to stream
 */
static void stream_weights(const float* src, hls::stream<weight_t>& ws, int n) {
    int pairs = n >> 1;
    for (int i = 0; i < pairs; i++) {
        #pragma HLS PIPELINE II=2
        ws.write((weight_t)src[2*i]);
        ws.write((weight_t)src[2*i + 1]);
    }
    if (n & 1)
        ws.write((weight_t)src[n-1]);
}

/**
 * @brief Performs a convolution layer (+RELU) with the given parameters.
 * 
 * @param w pointer to weights in DRAM
 * @param x pointer to input feature map in BRAM
 * @param b pointer to biases in BRAM
 * @param z pointer to output feature map in BRAM
 * @param Cin number of input channels
 * @param Cout number of output channels
 */
static void conv_layer(const act_t* x, const float* w, const bias_t* b,
                       act_t* z, int Cin, int Cout, int H, int W) {
    #pragma HLS DATAFLOW
    hls::stream<weight_t> weight_fifo;
    #pragma HLS STREAM variable=weight_fifo depth=256
    stream_weights(w, weight_fifo, Cout * Cin * 9);
    conv2d_core(x, weight_fifo, b, z, Cin, Cout, H, W);
}

static void maxpool_core(const act_t* in, act_t* out, int Cin, int H, int W) {
    int Ho = H >> 1, Wo = W >> 1;
    
    act_t line_buf[CONV_MAX_DIM >> 1];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    int in_ch = 0, out_ch = 0;                 
    for (int c = 0; c < Cin; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=CONV_MAX_CIN
        
        int in_row = in_ch;                    
        for (int y = 0; y < H; y++) {
            int out_row = out_ch + (y >> 1) * Wo;
            bool is_odd_row = (y & 1);

            //pre-read input => compute in next cycle
            act_t e1_reg = 0;
            act_t e2_reg = 0;

            
            for (int x = 0; x < W + 2; x += 2) {
                #pragma HLS PIPELINE II=1
                
                //Read inputs of next iteration
                act_t e1_next = 0;
                act_t e2_next = 0;
                if (x < W) {
                    e1_next = in[in_row + x];
                    e2_next = in[in_row + x + 1];
                }

                //Access previous reads to compute maxpool
                if (x > 0) {
                    act_t m = (e1_reg > e2_reg) ? e1_reg : e2_reg;
                    int ox = (x - 2) >> 1;
                    
                    if (is_odd_row) {
                        act_t prev = line_buf[ox];
                        out[out_row + ox] = (m > prev) ? m : prev;
                    } else {
                        line_buf[ox] = m;
                    }
                }

                //Save reads in registers for next iteration
                e1_reg = e1_next;
                e2_reg = e2_next;
            }
            in_row += W;                       
        }
        in_ch  += H * W;                       
        out_ch += Ho * Wo;
    }
}



static void avgpool_core(const act_t* in, act_t* out, int C, int H, int W) {
    int n = H * W;
    for (int c = 0; c < C; c++) {
        acc_t s = 0;
        for (int i = 0; i < n; i++)
            s += in[c*n + i];
        act_t v = (act_t)(s / n);
        out[c] = v;
    }
}


void convEngine(
    float *xin,
    float *wbuf,
    float *bbuf,
    float *zout
)
{
// HLS interface pragmas
#pragma HLS INTERFACE m_axi port=xin bundle=gmem0 depth=3072
#pragma HLS INTERFACE m_axi port=wbuf bundle=gmem1 depth=267696
#pragma HLS INTERFACE m_axi port=bbuf bundle=gmem2 depth=688
#pragma HLS INTERFACE m_axi port=zout bundle=gmem3 depth=64

#pragma HLS INTERFACE s_axilite port=xin  bundle=control
#pragma HLS INTERFACE s_axilite port=wbuf bundle=control
#pragma HLS INTERFACE s_axilite port=bbuf bundle=control
#pragma HLS INTERFACE s_axilite port=zout bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Ping-pong buffers for input/output feature maps (~16 BRAM36 blocks each)
    act_t fmap_bufA[ACT_MAX];
    act_t fmap_bufB[ACT_MAX];
    #pragma HLS bind_storage variable=fmap_bufA type=ram_2p impl=bram
    #pragma HLS bind_storage variable=fmap_bufB type=ram_2p impl=bram

    // Bias tile for the current layer (weights are streamed straight from DRAM)
    bias_t   bias_buf[BIAS_MAX];

    // Load the input image into buffer A
    load_dram<act_t>(xin, fmap_bufA, IMAGE_MAX);

    act_t* cur = fmap_bufA;   // current feature map
    act_t* nxt = fmap_bufB;   // scratch feature map
    int w_off = 0, b_off = 0;
    int out_size = IMAGE_MAX; // element count of the current feature map

    for (int L = 0; L < N_LAYERS; L++) {
        LayerParams lp = CNN_Layers[L];
        switch (lp.type) {

        case CONV: {
            //number of weights for this layer
            int wn = lp.Cout * lp.Cin * 9;

            //load biases from DRAM
            load_dram<bias_t>(bbuf + b_off, bias_buf, lp.Cout);

            //every conv in rnet20 is followed by ReLU -> always fuse it on write
            conv_layer(cur, wbuf + w_off, bias_buf, nxt,
                       lp.Cin, lp.Cout, lp.H, lp.W);

            //increment offsets for next layer's weights/biases
            w_off += wn;
            b_off += lp.Cout;

            //update the output size and swap ping-pong buffers
            out_size = lp.Cout * lp.H * lp.W;
            { act_t* t = cur; cur = nxt; nxt = t; }   // ping-pong swap
            break;
        }

        case MAXPOOL: {
            maxpool_core(cur, nxt, lp.Cin, lp.H, lp.W);
            out_size = lp.Cin * (lp.H >> 1) * (lp.W >> 1);
            { act_t* t = cur; cur = nxt; nxt = t; }
            break;
        }

        case AVGPOOL: {
            avgpool_core(cur, nxt, lp.Cin, lp.H, lp.W);
            out_size = lp.Cin;
            { act_t* t = cur; cur = nxt; nxt = t; }
            break;
        }

        case LINEAR:
        case SOFTMAX:
            // Done in software on the host. The kernel's final output is the
            // 64 avgpool features (out_size stays 64 from the AVGPOOL case).
            break;
        }
    }

    // Store the final result back to DRAM
    store_dram<act_t>(cur, zout, out_size);
}
