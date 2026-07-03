#include "hw_cnn.h"
#include "dtypes.h"
#include "conv2d/conv2d.h"

#define N_LAYERS 43

enum Layer_t { CONV, RELU, MAXPOOL, AVGPOOL, LINEAR, SOFTMAX };

struct LayerParams {
    Layer_t type;
    int Cin, Cout, H, W;
};

static const LayerParams CNN_Layers[N_LAYERS] = {
    {CONV, 3, 16, 32, 32},  {RELU, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32}, {RELU, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32}, {RELU, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32}, {RELU, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32}, {RELU, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32}, {RELU, 16, 16, 32, 32},
    {CONV, 16, 16, 32, 32}, {RELU, 16, 16, 32, 32},
    {MAXPOOL, 16, 16, 32, 32},                        
    {CONV, 16, 32, 16, 16}, {RELU, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16}, {RELU, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16}, {RELU, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16}, {RELU, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16}, {RELU, 32, 32, 16, 16},
    {CONV, 32, 32, 16, 16}, {RELU, 32, 32, 16, 16},
    {MAXPOOL, 32, 32, 16, 16},                        
    {CONV, 32, 64, 8, 8},   {RELU, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},   {RELU, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},   {RELU, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},   {RELU, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},   {RELU, 64, 64, 8, 8},
    {CONV, 64, 64, 8, 8},   {RELU, 64, 64, 8, 8},
    {AVGPOOL, 64, 64, 8, 8},                          
    {LINEAR, 64, 10, 1, 1},
    {SOFTMAX, 10, 10, 1, 1},
};


/**
 * @brief Copies n floats from off-chip DRAM (src) into an on-chip buffer (dst).
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
 * @brief In-place ReLU over n activations: x = max(x, 0). Own layer state.
 */
static void relu_core(act_t* x, int n) {
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        if (x[i] < (act_t)0) x[i] = (act_t)0;
    }
}

/**
 * @brief 2x2 stride-2 max pooling. Input C x H x W -> output C x H/2 x W/2.
 */
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

/**
 * @brief Global average pooling. Input C x H x W -> output C x 1 x 1.
 */
static void avgpool_core(const act_t* in, act_t* out, int C, int H, int W) {
    int n = H * W;
    for (int c = 0; c < C; c++) {
        acc_t s = 0;
        for (int i = 0; i < n; i++)
            s += in[c*n + i];
        out[c] = (act_t)(s / n);
    }
}

// NOTE: LINEAR + SOFTMAX are performed in software on the host (see
// CNN::inference). The kernel produces the 64 global-average-pool features;
// the tiny classifier runs on the ARM core, so linear weights never need to be
// streamed to the PL.


void convEngine(
    float *xin,
    float *wbuf,
    float *bbuf,
    float *zout
)
{
// HLS interface pragmas
#pragma HLS INTERFACE m_axi port=xin  offset=slave bundle=gmem0 depth=3072
#pragma HLS INTERFACE m_axi port=wbuf offset=slave bundle=gmem1 depth=267696
#pragma HLS INTERFACE m_axi port=bbuf offset=slave bundle=gmem2 depth=688
#pragma HLS INTERFACE m_axi port=zout offset=slave bundle=gmem3 depth=16384

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

    // Single (non-double-buffered) weight and bias tiles for the current layer
    weight_t weights_buf[WEIGHTS_MAX];
    bias_t   bias_buf[BIAS_MAX];

    // Load the input image into buffer A
    load_dram<act_t>(xin, fmap_bufA, IMAGE_MAX);

    act_t* cur = fmap_bufA;   // current feature map
    act_t* nxt = fmap_bufB;   // scratch feature map
    int w_off = 0, b_off = 0;
    int out_size = IMAGE_MAX; // element count of the current feature map

    // ---- layer state machine (rolled loop -> one time-shared engine) ----
    for (int L = 0; L < N_LAYERS; L++) {
        LayerParams lp = CNN_Layers[L];
        switch (lp.type) {

        case CONV: {
            //number of weights for this layer
            int wn = lp.Cout * lp.Cin * 9;  

            //load weights and biases from DRAM
            load_dram<weight_t>(wbuf + w_off, weights_buf, wn);
            load_dram<bias_t>(bbuf + b_off, bias_buf, lp.Cout);

            //compute the convolution for this layer
            conv2d_core(cur, weights_buf, bias_buf, nxt,
                        lp.Cin, lp.Cout, lp.H, lp.W);

            //increment offsets for next layer's weights/biases
            w_off += wn;
            b_off += lp.Cout;
            
            //update the output size and swap ping-pong buffers
            out_size = lp.Cout * lp.H * lp.W;
            { act_t* t = cur; cur = nxt; nxt = t; }   // ping-pong swap
            break;
        }

        case RELU:
            relu_core(cur, lp.Cout * lp.H * lp.W);     // in place, no swap
            break;

        case MAXPOOL:
            maxpool_core(cur, nxt, lp.Cin, lp.H, lp.W);
            out_size = lp.Cin * (lp.H >> 1) * (lp.W >> 1);
            { act_t* t = cur; cur = nxt; nxt = t; }
            break;

        case AVGPOOL:
            avgpool_core(cur, nxt, lp.Cin, lp.H, lp.W);
            out_size = lp.Cin;
            { act_t* t = cur; cur = nxt; nxt = t; }
            break;

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
