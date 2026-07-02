#include "hw_cnn.h"



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

    //TODO: implement CNN pipeline here
}
