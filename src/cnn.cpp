#include "cnn.h"
#ifdef _WIN32
#include <chrono>
#endif

namespace ml {

CNN::CNN(std::vector<CNN_layer_struct> in_layers)
{
	uint32_t insize = 0;
	layers = in_layers;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU: case Layer_Type::Softmax:
				lay.output_size[0] = layers[i - 1].output_size[0];
				lay.output_size[1] = layers[i - 1].output_size[1];
				lay.output_size[2] = layers[i - 1].output_size[2];
				if(lay.in_place)
					lay.Z = layers[i - 1].Z;
				else
					lay.Z = new Tensor(layers[i-1].output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::MaxPool:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::AvgPool:
				lay.Z = new Tensor(lay.output_size[0],1,1);
				break;
			case Layer_Type::Conv:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				lay.W = new Tensor[lay.output_size[0]]();
				for(int i =0 ; i < lay.output_size[0]; i++){
					lay.W[i].allocate(lay.input_channels,lay.kernel_width,lay.kernel_width);
				}
				lay.B = new Tensor(1,1,lay.output_size[0]);
				break;
			case Layer_Type::Linear:
				insize = layers[i-1].output_size[0] * layers[i-1].output_size[1] * layers[i-1].output_size[2];
				lay.Z = new Tensor(1,1,lay.output_size[2]);
				lay.W = new Tensor(1,lay.output_size[2],insize);
				lay.B = new Tensor(1,1,lay.output_size[2]);
				break;
			default:
				throw std::runtime_error("Layer not implemented !\n");
		}
	}
}



CNN::~CNN()
{
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU:
				if(!(lay.in_place))
					delete lay.Z;
				break;
			case Layer_Type::Softmax:
			case Layer_Type::MaxPool:
				delete lay.Z;
				break;
			case Layer_Type::AvgPool:
				delete lay.Z;
				break;
			case Layer_Type::Conv:
				delete lay.Z;
				delete [] lay.W;
				delete lay.B;
				break;
			case Layer_Type::Linear:
				delete lay.Z;
				delete lay.W;
				delete lay.B;
				break;
			default:
				printf("Rogue unimplemented layer found during deallocation !\n");
		}
	}
}



/* Implement Inference here !*/
void CNN::inference(Tensor * input, int N, uint8_t preds[])
{
#ifdef HLS
	int w_total,b_total;
	FLOAT * w_all = genSeqConvWeights(&w_total);
	FLOAT * b_all = genSeqConvBias(&b_total);

	for(int iter = 0; iter < N; iter++){
		Tensor * X = &(input[iter]);
		/* Insert your Board Code here */


		X = layers[42].Z;
		/* Store your output in tensor X->data[0][0] */
		/* Get the max value from tensor */
		float max = 0;
		uint8_t pred = 0;
		for(int k = 0; k < X->size[2]; k++){
			const float v = (*X)[0][0][k];
			if(v > max){
				pred =k;
				max = v;
			}
		}
		preds[iter] = pred;
	}
	delete [] w_all;
	delete [] b_all;
#elif defined BOARD
	int w_total,b_total;
	FLOAT * w_all = genSeqConvWeights(&w_total);
	FLOAT * b_all = genSeqConvBias(&b_total);

	for(int iter = 0; iter < N; iter++){
		Tensor * X = &(input[iter]);
		/* Insert your Board Code here */



		X = layers[42].Z;
		/* Store your output in tensor X->data[0][0] */
		/* Get the max value from tensor */
		float max = 0;
		uint8_t pred = 0;
		for(int k = 0; k < X->size[2]; k++){
			const float v = (*X)[0][0][k];
			if(v > max){
				pred =k;
				max = v;
			}
		}
		preds[iter] = pred;
	}
	delete [] w_all;
	delete [] b_all;
#else
	for(int iter = 0; iter < N; iter++){
		Tensor * X = &(input[iter]);
		for(int i = 0; i < layers.size(); i++){
			CNN_layer_struct * lay = &(layers[i]);
#ifndef _WIN32
				auto start = mtick();
#else
				auto start = std::chrono::high_resolution_clock::now();
#endif
			switch(lay->type){
				case Layer_Type::Conv:
					if(lay->pad > 0)
						X = padTensor(X,lay->pad);
					conv2d(X,lay->W, lay->B,lay->Z);
					if(lay->pad > 0)
						delete X;
					break;
				case Layer_Type::ReLU:
					ReLU(X,lay->Z);
					break;
				case Layer_Type::MaxPool:
					maxPool(X,lay->Z);
					break;
				case Layer_Type::Linear:
					Linear(X,lay->W,lay->B,lay->Z);
					break;
				case Layer_Type::Softmax:
					Softmax(X,lay->Z);
					break;
				case Layer_Type::AvgPool:
					avgPool(X,lay->Z);
					break;
				default:
					printf("Unimplemented Layer %d !\n",(uint32_t) lay->type);
					return;
				}
#ifndef _WIN32
				double time = mtock(start);
				this->runtime[(uint32_t) lay->type] += time;
#else
				auto stop = std::chrono::high_resolution_clock::now();
				double time = std::chrono::duration<double, std::milli>(stop - start).count();
				this->runtime[(uint32_t) lay->type] += time;
#endif
			X= lay->Z;
		}
		/* Get the max value from tensor */
		float max = 0;
		uint8_t pred = 0;
		for(int k = 0; k < X->size[2]; k++){
			const float v = (*X)[0][0][k];
			if(v > max){
				pred =k;
				max = v;
			}
		}
		preds[iter] = pred;
	}
#endif
}


}
