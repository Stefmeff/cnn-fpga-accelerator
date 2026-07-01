#ifndef HW_CNN_H
#define HW_CNN_H

#include "stdint.h"

// Simple top-level kernel declaration for Vitis HLS.
// Keep signature compatible with host example: krnl(xin,wbuf,bbuf,zout)
extern "C" void convEngine(float *xin, float *wbuf, float *bbuf, float *zout);

#endif
