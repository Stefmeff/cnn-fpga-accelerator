#include "hw_cnn.h"
#include "dtypes.h"
#include "ap_int.h"


/**
 * @brief loads the input feature map (3x32x32 image) from off-chip DRAM 
 * sinto the on-chip BRAM-buffer
 * 
 * @param src axi memory access to load image from
 * @param dst on-chip buffer to store the feature map
 * @param N number of elements to load
 */
void load_fmap(const float* src, float* dst, ap_uint<12> N) {
    for (ap_uint<12> i = 0; i < N; i++) { 
        #pragma HLS PIPELINE II=2
        dst[i*2] = src[i*2];  
        dst[i*2+1] = src[i*2+1];            
    }
}

void load_weights(const float* src, float* dst, ap_uint<12> N) {
    for (ap_uint<12> i = 0; i < N; i++) { 
        #pragma HLS PIPELINE II=2
        dst[i*2] = src[i*2];  
        dst[i*2+1] = src[i*2+1];            
    }
}

void load_biases(const float* src, float* dst, ap_uint<12> N) {
    for (ap_uint<12> i = 0; i < N; i++) { 
        #pragma HLS PIPELINE II=2
        dst[i*2] = src[i*2];  
        dst[i*2+1] = src[i*2+1];            
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
#pragma HLS INTERFACE m_axi port=xin  offset=slave bundle=gmem0 depth=1
#pragma HLS INTERFACE m_axi port=wbuf offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE m_axi port=bbuf offset=slave bundle=gmem2 depth=1
#pragma HLS INTERFACE m_axi port=zout offset=slave bundle=gmem3 depth=1

#pragma HLS INTERFACE s_axilite port=xin  bundle=control
#pragma HLS INTERFACE s_axilite port=wbuf bundle=control
#pragma HLS INTERFACE s_axilite port=bbuf bundle=control
#pragma HLS INTERFACE s_axilite port=zout bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    //Ping-pong buffers for input and output feature maps
    act_t fmap_bufA[ACT_MAX];
    act_t fmap_bufB[ACT_MAX];
    #pragma HLS bind_storage variable=bufA type=ram_2p impl=bram
    #pragma HLS bind_storage variable=bufB type=ram_2p impl=bram

    //Buffers for weights and biases
    weight_t weights_buf[WEIGHTS_MAX];
    bias_t bias_buf[BIAS_MAX];

    load_fmap(xin, fmap_bufA, (ap_uint<12>)IMAGE_MAX);
}
