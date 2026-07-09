
#include "cnn.h"

#define HLS

#ifdef BOARD
#include <xrt.h>
#include "bitLoad.h"
#endif

#ifdef HLS
#include "../hls/hw_cnn.h"   
#define HLS                  
#endif

#ifdef _WIN32
#include <chrono>
#endif


#ifndef XCLBIN_PATH
#define XCLBIN_PATH "./kernel/convEngine2.xclbin"
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
	FLOAT * w_all = genSeqConvWeightsWS(&w_total);   // conv weights only
	FLOAT * b_all = genSeqConvBias(&b_total);

	CNN_layer_struct * lin = &layers[41];          // LINEAR layer (run in software)

	FLOAT zout[64];                                // 64 global-average-pool features
	Tensor avg(64, 1, 1);
	Tensor logits(1, 1, 10);
	Tensor probs(1, 1, 10);

	for(int iter = 0; iter < N; iter++){
		Tensor * X = &(input[iter]);

		
		convEngine(X->data[0][0], w_all, b_all, zout);


		memcpy(avg.data[0][0], zout, sizeof(FLOAT) * 64);
		Linear(&avg, lin->W, lin->B, &logits);
		Softmax(&logits, &probs);

		memcpy(layers[42].Z->data[0][0], probs.data[0][0], sizeof(FLOAT) * 10);

		// argmax over the 10 class scores (softmax preserves ordering)
		float max = probs.data[0][0][0];
		uint8_t pred = 0;
		for(int k = 1; k < 10; k++){
			const float v = probs.data[0][0][k];
			if(v > max){ max = v; pred = (uint8_t)k; }
		}
		preds[iter] = pred;
	}
	delete [] w_all;
	delete [] b_all;
#elif defined BOARD
	//generate weights and biase into flattened array
	int w_total,b_total;
	FLOAT * w_all = genSeqConvWeightsWS(&w_total);   // all conv weights, contiguous
	FLOAT * b_all = genSeqConvBias(&b_total);      // all conv biases,  contiguous

	//load the FPGA bitstream and create the kernel object
	auto device = xrt::device(0);
	auto xclbin_uuid = device.load_xclbin(XCLBIN_PATH);
	loadBitstream(XCLBIN_PATH);                    // program the PL (helper from the edge-detection lab)
	auto krnl = xrt::kernel(device, xclbin_uuid, "convEngine2",
	                        xrt::kernel::cu_access_mode::exclusive);

	//initialize buffers for fmap, weights, biases, and output
	auto xin  = xrt::bo(device, sizeof(float) * 3 * 32 * 32, krnl.group_id(0));
	auto wbuf = xrt::bo(device, sizeof(float) * w_total,     krnl.group_id(1));
	auto bbuf = xrt::bo(device, sizeof(float) * b_total,     krnl.group_id(2));
	auto zout = xrt::bo(device, sizeof(float) * 64,          krnl.group_id(3));

	auto start = mtick();

	//write weight and biases to buffer
	wbuf.write(w_all);
	double w_time = mtock(start);

	start = mtick();
	bbuf.write(b_all);
	double b_time = mtock(start);
	printf("---------------------------\n");
	printf("Weights write time[ms]: %lf\n",w_time);
	printf("Biases write time[ms]: %lf\n",b_time);


	CNN_layer_struct * lin = &layers[41];          
	Tensor avg(64, 1, 1);                         
	Tensor logits(1, 1, 10);
	Tensor probs(1, 1, 10);

	start = mtick();
	for(int iter = 0; iter < N; iter++){
		//write inpput tensor to buffer
		double start2 = mtick();
		Tensor * X = &(input[iter]);
		xin.write(X->data[0][0]);
		double i_write = mtock(start2);


		//run and measure kernel execution time
		start2 = mtick();
		auto run = krnl(xin, wbuf, bbuf, zout);    // start kernel with all 4 args
		run.wait();                                // block until the IP is done
		double kernel_time = mtock(start2);

		start2 = mtick();
		zout.read(avg.data[0][0]);
		double o_read = mtock(start2);

		start2 = mtick();
		Linear(&avg, lin->W, lin->B, &logits);
		Softmax(&logits, &probs);
		double linear_softmax_time = mtock(start2);

		memcpy(layers[42].Z->data[0][0], probs.data[0][0], sizeof(FLOAT) * 10);

		start2 = mtick();
		float max = probs.data[0][0][0];
		uint8_t pred = 0;
		for(int k = 1; k < 10; k++){
			const float v = probs.data[0][0][k];
			if(v > max){ max = v; pred = (uint8_t)k; }
		}
		preds[iter] = pred;
		double pred_time = mtock(start2);

		printf("---------------------------\n");
		printf("IMAGE NUMBER: %d\n",iter);
		printf("Image write time[ms]: %lf\n",i_write);
		printf("Kernel time[ms]: %lf\n",kernel_time);
		printf("Output read time[ms]: %lf\n",o_read);
		printf("Linear + Softmax time[ms]: %lf\n",linear_softmax_time);
		printf("Argmax time[ms]: %lf\n",pred_time);
		printf("---------------------------\n");
	}
	double total_time = mtock(start);
	printf("Total inference time[ms]: %lf\n",total_time);
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
