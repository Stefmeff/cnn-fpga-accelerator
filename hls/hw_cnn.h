#ifndef HW_CNN_H
#define HW_CNN_H

#include "stdint.h"

/**
 * @brief Kernel function for the CNN accelerator.
 * 
 * @param xin Input feature map (image 3x32x32)
 * @param wbuf weights 
 * @param bbuf biases
 * @param zout Output feature map 
 */
void convEngine(
    float *xin, 
    float *wbuf, 
    float *bbuf, 
    float *zout
);

#endif
