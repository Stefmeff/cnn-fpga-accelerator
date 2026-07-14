#include "cnn.h"
#include "../hls/hw_cnn.h"

namespace ml {

int checkLayerMeta(CNN_layer_struct * lay, FILE * f)
{
	CNN_layer_struct tmp;
	if(fread(&(tmp.type),				sizeof(tmp.type),		1,f) == 0){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.input_channels),	sizeof(tmp.input_channels),1,f) ==0){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.output_size[0]),			sizeof(tmp.output_size[0]),	1,f) == 0 ){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.kernel_width),		sizeof(tmp.kernel_width),	1,f) == 0 ){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.pad),				sizeof(tmp.pad),		1,f) == 0 ){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	// Check correctness
	if(!(tmp.type == lay->type))
		return CNN_RETURN_FAILED;
	if(lay->type == Layer_Type::Conv){
		if(((tmp.input_channels == lay->input_channels) &&
			(tmp.kernel_width == lay->kernel_width) &&
			(tmp.pad == lay->pad) &&
			(tmp.output_size[0] == lay->output_size[0])) == 0){
			return CNN_RETURN_FAILED;
		}
	}
	if(lay->type == Layer_Type::Linear){
		if(!(tmp.output_size[0] == tmp.output_size[0]))
			return CNN_RETURN_FAILED;
	}
	return CNN_RETURN_SUCCESS;
}


int readConvWeights(CNN_layer_struct * lay, FILE * f)
{
	int res;
	for(int i = 0; i < lay->output_size[0]; i++){
		Tensor * W = &(lay->W[i]);
		if(res = W->read(f) ){
			printf("Weight tensor read has failed code %d!\n", res);
			return CNN_RETURN_FAILED;
		}
	}
	Tensor * B = lay->B;
	if(res = B->read(f)){
		printf("Bias tensor read has failed code %d!\n", res);
		return CNN_RETURN_FAILED;
	}
	return CNN_RETURN_SUCCESS;
}



int readFCWeights(CNN_layer_struct * lay, uint32_t insize, FILE * f)
{
	int res;
	Tensor * W = lay->W;
	if(res = W->read(f)){
		printf("Linear Weight Tensor read failed code %d \n!",res);
		return CNN_RETURN_FAILED;
	}
	Tensor * B = lay->B;
	if(res = B->read(f)){
		printf("Linear Bias Tensor read failed code %d \n!",res);
		return CNN_RETURN_FAILED;
	}
	return CNN_RETURN_SUCCESS;
}






int CNN::read(const char * infile)
{
	FILE * f = fopen(infile,"rb");
	uint32_t nlayers;
	if(fread(&(nlayers),sizeof(nlayers),1,f) == 0){
		printf("Reading number of layers failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(layers.size() != nlayers){
		printf("Layer sizes do not match reading weights failed!\n");
		return CNN_RETURN_FAILED;
	}
	uint32_t prev_size = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(checkLayerMeta(lay,f) == CNN_RETURN_FAILED){
			printf("Layer check failed!\n");
			return CNN_RETURN_FAILED;
		} 
		switch(lay->type){ 
			case Layer_Type::MaxPool:
			case Layer_Type::AvgPool:
			case Layer_Type::ReLU:
			case Layer_Type::Softmax:
				break;
			case Layer_Type::Conv:
				if(readConvWeights(lay,f) == CNN_RETURN_FAILED)
					return CNN_RETURN_FAILED;
				break;
			case Layer_Type::Linear:
				if(readFCWeights(lay,prev_size,f) == CNN_RETURN_FAILED)
					return CNN_RETURN_FAILED;
				break;
			default:
				printf("%d not a valid layer type!\n",(uint32_t) lay->type);
		}
		prev_size = lay->output_size[2] * lay->output_size[1] * lay->output_size[0];
	}
	fclose(f);
	return CNN_RETURN_SUCCESS;
}


void CNN::print_timing(int N)
{
	const char * names[] = {"Linear","Pool","ReLU","Conv","Softmax","GlobalAvgPool"};
	printf("\n--------------------\n");
	printf("Avg Execution Time [ms]:\n");
	printf("--------------------\n");
	double total = 0.0;
	for(int i = 0; i < 6; i++){
		printf("%s Layer: %lf \n",names[i],runtime[i]/N);
		total+= runtime[i];
	}
	printf("--------------------\n");
}

/* ----------------------------- Functions to initialize Layers -----------------------------*/

CNN_layer_struct LinearLayer(uint32_t outputs)
{
	CNN_layer_struct linear;
	linear.type = Layer_Type::Linear;
	linear.output_size[0] = 1;
	linear.output_size[1] = 1;
	linear.output_size[2] = outputs;
	return linear;
}

CNN_layer_struct MaxPoolLayer(uint32_t output_channels, uint32_t output_height, uint32_t output_width)
{
	CNN_layer_struct pool;
	pool.type = Layer_Type::MaxPool;
	pool.output_size[0] = output_channels;
	pool.output_size[1] = output_height;
	pool.output_size[2] = output_width;
	return pool;
}


CNN_layer_struct AvgPoolLayer(uint32_t output_channels)
{
	CNN_layer_struct pool;
	pool.type = Layer_Type::AvgPool;
	pool.output_size[0] = output_channels;
	pool.output_size[1] = 1;
	pool.output_size[2] = 1;
	return pool;
}


CNN_layer_struct ConvLayer(uint32_t input_channels, uint32_t output_channels, uint32_t output_height, 
		uint32_t output_width, uint32_t kernel_width, uint32_t pad)
{
	CNN_layer_struct conv;
	conv.type = Layer_Type::Conv;
	conv.kernel_width = kernel_width;
	conv.pad = pad;
	conv.output_size[0] = output_channels;
	conv.output_size[1] = output_height;
	conv.output_size[2] = output_width;
	conv.input_channels = input_channels;
	return conv;
}


CNN_layer_struct ReLULayer(bool in_place )
{
	CNN_layer_struct relu;
	relu.type = Layer_Type::ReLU;
	relu.in_place = in_place;
	return relu;
}

CNN_layer_struct SoftmaxLayer()
{
	CNN_layer_struct lay;
	lay.type = Layer_Type::Softmax;
	lay.in_place = false;
	return lay;
}

FLOAT * CNN::genSeqConvWeights(int * w_total)
{
	*w_total = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(lay->type == Layer_Type::Conv){
			*w_total += lay->output_size[0]*lay->input_channels*9;
		}
	}
	printf("Allocating %d floats!\n",*w_total);
	FLOAT * w_all = new FLOAT[*w_total] ;
	if(w_all == NULL){
		printf("ERROR: Failed Allocation!\n");
		return 0;
	}
	int w_off = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(lay->type == Layer_Type::Conv){
			int w_size = lay->output_size[0]*lay->input_channels*9;
			for(int oc = 0; oc < lay->output_size[0]; oc++){
				memcpy(w_all + w_off,lay->W[oc].data[0][0],
						sizeof(FLOAT)*lay->input_channels*9);
				w_off += lay->input_channels*9;
			}
		}
	}
	return w_all;
}

// Change weight streaming order => optimized for weight-stationary apporach
//Conv weights in convEngine consumption order: per layer, oc0-tile -> ic -> t -> ky -> kx.
FLOAT * CNN::genSeqConvWeightsWS(int * w_total)
{
	const int T = COUT_TILE;

	*w_total = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(lay->type == Layer_Type::Conv)
			*w_total += lay->output_size[0]*lay->input_channels*9;
	}

	printf("Allocating %d floats (WS order, T=%d)!\n",*w_total,T);
	FLOAT * w_all = new FLOAT[*w_total];
	if(w_all == NULL){
		printf("ERROR: Failed Allocation!\n");
		return 0;
	}

	int w_off = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(lay->type != Layer_Type::Conv) continue;

		const int Cin  = lay->input_channels;
		const int Cout = lay->output_size[0];
		if(Cout % T != 0)
			printf("WARNING: layer %d Cout=%d not divisible by COUT_TILE=%d!\n",i,Cout,T);

		// Packed-beat order: [oc0][ic][ky][kx][t]. The T (=COUT_TILE) lanes are
		// innermost so 16 consecutive floats form one fvec beat carrying all
		// output-channel lanes of a single kernel column (matches init_kernel).
		for(int oc0 = 0; oc0 < Cout; oc0 += T)
			for(int ic = 0; ic < Cin; ic++)
				for(int ky = 0; ky < 3; ky++)
					for(int kx = 0; kx < 3; kx++)
						for(int t = 0; t < T; t++){
							int oc = oc0 + t;
							w_all[w_off++] = lay->W[oc].data[ic][ky][kx];
						}
	}
	return w_all;
}

FLOAT * CNN::genSeqConvBias(int * b_total)
{
	*b_total = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(lay->type == Layer_Type::Conv){
			*b_total += lay->output_size[0];
		}
	}
	printf("Bias: Allocating %d floats!\n",*b_total);
	FLOAT * b_all = new FLOAT[*b_total];
	if(b_all == NULL){
		printf("ERROR: Failed Allocation!\n");
		return 0;
	}
	int b_off = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(lay->type == Layer_Type::Conv){
			memcpy(b_all + b_off,lay->B->data[0][0],sizeof(FLOAT)*lay->output_size[0]);
			b_off += lay->output_size[0];
		}
	}

	return b_all;
}

}
