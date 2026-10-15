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

//streams weights from DRAM to a FIFO => each read returns a wtile (COUT_TILE weights) for one (ic,ky,kx)
static void stream_weights(const fvec* src, hls::stream<wtile>& ws, int nwords) {
    for (int i = 0; i < nwords; i++) {
        #pragma HLS PIPELINE II=1
        fvec beat = src[i];
        wtile col;
        //pack COUT_TILE weights into a single FIFO word => only 9 reads per (tile,ic) instead of COUT_TILE*9
        for (int t = 0; t < COUT_TILE; t++) {
            #pragma HLS UNROLL
            col.w[t] = (weight_t)beat.v[t];
        }
        ws.write(col);
    }
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
static void conv_layer(const act_t* x, const fvec* w, const bias_t* b,
                       act_t* z, int Cin, int Cout, int H, int W) {
    // One packed beat per (oc0-tile, ic, ky, kx); each carries COUT_TILE lanes.
    int total_words = (Cout / COUT_TILE) * Cin * 9;
    #pragma HLS DATAFLOW

    hls::stream<wtile> weight_fifo;
    #pragma HLS STREAM variable=weight_fifo depth=32
    #pragma HLS bind_storage variable=weight_fifo type=fifo impl=srl
    stream_weights(w, weight_fifo, total_words);
    conv2d_ws(x, weight_fifo, b, z, Cin, Cout, H, W);
}

 
static void maxpool_core(const act_t* in, act_t* out, int C, int H, int W) {
    int Ho = H >> 1, Wo = W >> 1;
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                int iy = oy << 1, ix = ox << 1;
                act_t m = in[(c*H + iy)*W + ix];
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        act_t v = in[(c*H + iy + dy)*W + ix + dx];
                        if (v > m) m = v;
                    }
                out[(c*Ho + oy)*Wo + ox] = m;
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
    const fvec *wbuf,
    float *bbuf,
    float *zout
)
{
#pragma HLS INTERFACE m_axi port=xin offset=slave bundle=gmem0 depth=3072
#pragma HLS INTERFACE m_axi port=wbuf offset=slave bundle=gmem1 depth=16731 max_read_burst_length=16 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=bbuf offset=slave bundle=gmem2 depth=688
#pragma HLS INTERFACE m_axi port=zout offset=slave bundle=gmem3 depth=64

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

    bias_t   bias_buf[BIAS_MAX];

    // Load the input image into buffer A
    load_dram<act_t>(xin, fmap_bufA, IMAGE_MAX);

    act_t* cur = fmap_bufA;   // current feature map
    act_t* nxt = fmap_bufB;   // scratch feature map
    int w_off = 0, b_off = 0;
    int out_size = IMAGE_MAX; // element count of the current feature map
    int ping_pong_state = 0; 


    for (int L = 0; L < N_LAYERS; L++) {
        LayerParams lp = CNN_Layers[L];
        switch (lp.type) {

        case CONV: {
            //number of packed weight beats for this layer (COUT_TILE weights each)
            int wn = (lp.Cout / COUT_TILE) * lp.Cin * 9;

            //load biases from DRAM
            load_dram<bias_t>(bbuf + b_off, bias_buf, lp.Cout);

            //run convolutional layer
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
            // Done in software on the host.
            break;
        }
    }

    // Store the final result back to DRAM
    store_dram<act_t>(cur, zout, out_size);
}