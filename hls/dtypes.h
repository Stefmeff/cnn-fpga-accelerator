/**
 * @file dtypes.h
 * 
 * @brief defines the data types used accross the project.
 * 
 */
#ifndef DTYPES_H
#define DTYPES_H

//#define USE_FIXED_POINT

#ifdef USE_FIXED_POINT
    #include <ap_fixed.h>
    typedef ap_fixed<16, 8>  act_t;    // activations / feature maps   (widths TBD)
    typedef ap_fixed<16, 2>  weight_t; // weights
    typedef ap_fixed<16, 8>  bias_t;   // biases
    typedef ap_fixed<32,16>  acc_t;    // accumulator — WIDER, avoids overflow
#else

    typedef float act_t;
    typedef float weight_t;
    typedef float bias_t;
    typedef float acc_t;
#endif
#endif
