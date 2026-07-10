/**
 * @file hw_cnn.h
 * @author Stefan Moser
 * @brief Top-level interface for the CNN accelerator kernel.
 * 
 */

#ifndef HW_CNN_H
#define HW_CNN_H

#include "stdint.h"

#define ACT_MAX 16384      //maximum size of output feature map (16×32×32)
#define WEIGHTS_MAX 36864    // 64×64×9
#define BIAS_MAX 64

#define IMAGE_MAX 3072


#define COUT_TILE 16

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
