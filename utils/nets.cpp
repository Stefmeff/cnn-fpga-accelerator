#include "nets.h"

namespace ml{

std::vector<CNN_layer_struct> rnet20 = {
	// Initial Conv
	ConvLayer(3,16,32,32,3,1),
	ReLULayer(),
	// Block 1 (32x32)
	ConvLayer(16,16,32,32,3,1),
	ReLULayer(),
	ConvLayer(16,16,32,32,3,1),
	ReLULayer(),
	ConvLayer(16,16,32,32,3,1),
	ReLULayer(),
	ConvLayer(16,16,32,32,3,1),
	ReLULayer(),
	ConvLayer(16,16,32,32,3,1),
	ReLULayer(),
	ConvLayer(16,16,32,32,3,1),
	ReLULayer(),
	// Pool
	MaxPoolLayer(16,16,16),
	// Block 2 (16x16)
	ConvLayer(16,32,16,16,3,1),
	ReLULayer(),
	ConvLayer(32,32,16,16,3,1),
	ReLULayer(),
	ConvLayer(32,32,16,16,3,1),
	ReLULayer(),
	ConvLayer(32,32,16,16,3,1),
	ReLULayer(),
	ConvLayer(32,32,16,16,3,1),
	ReLULayer(),
	ConvLayer(32,32,16,16,3,1),
	ReLULayer(),
	// Pool
	MaxPoolLayer(32,8,8),
	// Block 3 (16x16)
	ConvLayer(32,64,8,8,3,1),
	ReLULayer(),
	ConvLayer(64,64,8,8,3,1),
	ReLULayer(),
	ConvLayer(64,64,8,8,3,1),
	ReLULayer(),
	ConvLayer(64,64,8,8,3,1),
	ReLULayer(),
	ConvLayer(64,64,8,8,3,1),
	ReLULayer(),
	ConvLayer(64,64,8,8,3,1),
	ReLULayer(),
	// Classifier
	AvgPoolLayer(64),
	LinearLayer(10),
	SoftmaxLayer(),
};

}
