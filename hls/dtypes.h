/**
 * @file dtypes.h
 * 
 * @brief defines the data types used accross the project.
 * 
 */
#ifndef DTYPES_H
#define DTYPES_H

#ifndef USE_FLOAT         
#define USE_FIXED_POINT
#endif


#ifndef FP_ACT_W
  #define FP_ACT_W 16
  #define FP_ACT_I 8      
  #define FP_W_W   16
  #define FP_W_I   4      
  #define FP_ACC_W 32
  #define FP_ACC_I 16
#endif

#ifdef USE_FIXED_POINT
    #include <ap_fixed.h>
    typedef ap_fixed<FP_ACT_W, FP_ACT_I>  act_t;    // activations / feature maps
    typedef ap_fixed<FP_W_W,   FP_W_I>    weight_t; // weights
    typedef ap_fixed<FP_ACT_W, FP_ACT_I>  bias_t;   // biases (same range as act)
    typedef ap_fixed<FP_ACC_W, FP_ACC_I>  acc_t;
#else

    typedef float act_t;
    typedef float weight_t;
    typedef float bias_t;
    typedef float acc_t;
#endif
#endif
