#include "hw_cnn.h"

// Basic HLS includes (Vitis HLS)
#ifdef __SDSCC__
#include <sds_lib.h>
#endif

// NOTE: Implement the kernel body below. This skeleton follows the
// Edge Detection lab conventions: use m_axi for big buffers and s_axilite
// for control (implicit for top function).

extern "C" void convEngine(float *xin, float *wbuf, float *bbuf, float *zout)
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

    // Minimal skeleton: host will pack inputs and weights using
    // CNN::genSeqConvWeights()/genSeqConvBias() and pass pointers here.

    // Implementation plan:
    // - Load (burst) weights/biases into on-chip buffers if needed
    // - Stream input tiles from xin via local buffers
    // - Perform conv compute and write to zout
    // Use DATAFLOW, PIPELINE, and array_partition pragmas where appropriate.

    // For now: pass-through (placeholder) — copy input to output (for build/test)
    // Host must ensure sizes; this is only a placeholder to check integration.
    // TODO: implement actual convolution engine.

    // Example simple copy loop (replace with compute pipeline):
    // NOTE: depth pragmas above are placeholders; adjust according to actual sizes.
    for (int i = 0; i < 1; ++i) {
        // nothing by default
    }
}
