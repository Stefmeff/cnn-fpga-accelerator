#ifndef CNN_H
#define CNN_H


#define LAYER_TYPE_FC 		0
#define LAYER_TYPE_POOL		1
#define LAYER_TYPE_RELU		2
#define LAYER_TYPE_CONV		3
#define LAYER_TYPE_SOFTMAX	4
#define LAYER_TYPE_AVGPOOL	5


#define CNN_RETURN_SUCCESS 0
#define CNN_RETURN_FAILED 1

#include "tensor.h"
#include "kernels.h"
#include "stdint.h"
#include <vector>
#include <cmath>
#include <cstring>


#ifdef HLS
// Include your HLS HEADER

#elif defined BOARD
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>
#include "bitLoad.h"
#endif

namespace ml{

enum struct Layer_Type : uint32_t {Linear = 0,MaxPool = 1 , ReLU = 2,Conv = 3,Softmax = 4, AvgPool = 5};


typedef struct{
	Layer_Type type;
	uint32_t output_size[3];
	uint32_t kernel_width;
	uint32_t input_channels;
	uint32_t pad;
	bool in_place = false;
	Tensor * Z;
	Tensor * W;
	Tensor * B;
} CNN_layer_struct;


class CNN{
	public:
	double runtime[6] = {0};
	std::vector<CNN_layer_struct> layers;
	CNN(std::vector<CNN_layer_struct> in_layers);
	~CNN();
	int read(const char * infile);
	void inference(Tensor * inputs, int N, uint8_t preds[]);
	void print_timing(int N);
	FLOAT * genSeqConvWeights(int * w_total);
	FLOAT * genSeqConvWeightsWS(int * w_total);   // weight-stationary stream order
	FLOAT * genSeqConvBias(int * b_total);
};

/* ----------------------------- Functions to initialize Layers -----------------------------*/

CNN_layer_struct LinearLayer(uint32_t outputs);

CNN_layer_struct MaxPoolLayer(uint32_t output_channels, uint32_t output_height, uint32_t output_width);


CNN_layer_struct AvgPoolLayer(uint32_t output_channels);

CNN_layer_struct ConvLayer(uint32_t input_channels, uint32_t output_channels, uint32_t output_height, 
		uint32_t output_width, uint32_t kernel_width, uint32_t pad);

CNN_layer_struct ReLULayer(bool in_place = true);

CNN_layer_struct SoftmaxLayer();


}


#endif
