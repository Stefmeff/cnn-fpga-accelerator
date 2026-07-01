#ifndef BMP_2_TENSOR
#define BMP_2_TENSOR

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include "tensor.h"


const float norm_mean[3] = {0.485, 0.456, 0.406};
const float norm_std[3] = {0.229, 0.224, 0.225};

/*
 * read BMP file and normalize using the above mean and std deviation:
 */
Tensor * readBMP(const char * infile);

#endif
