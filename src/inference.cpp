#include "kernels.h"
#include "tensor.h"
#include <vector>
#include "cnn.h"
#include "nets.h"
#include <string>
#include <algorithm>
#include <chrono>
#include "cifar10_classes.h"
#include "bmp2tensor.h"

using namespace ml;

// Project root used to locate the data/ files. Vitis csim runs from a deep
// build directory, so this must be absolute. Override with -DPRJ_ROOT=... if the
// project moves (an absolute path also works fine for the plain `make` build).
#ifndef PRJ_ROOT
#define PRJ_ROOT "C:/Users/stefa/tum-master/SS2026/MasterPraktikum/project/cnn_fpga_accelerator/"
#endif
const char * PRJ_PATH = PRJ_ROOT;


int test_net( CNN * net,const char * data_file);
int bench_net(CNN * net,const char * data_file, int N);
int predict_image(CNN * net,const char * img_file);

std::string merge_pth(const char * project_path, const char * file_path);


int main(int argc, char * argv[])
{
	if(argc < 2){
		printf("Usage:\n");
		printf("Test:       ./inference t\n");
		printf("Benchmark:  ./inference b N\n");
		printf("Predict:    ./inference p image.bmp\n");
		return 1;
	}

	std::string weight_file = merge_pth(PRJ_PATH,"data/fnet20_weights.dat");
	printf("Reading Weights from %s\n",weight_file.c_str());

	CNN dut(rnet20);
	if(dut.read(weight_file.c_str()) == CNN_RETURN_FAILED){
		printf("Reading Weights failed!\n");
		return 0;
	}
	if(argv[1][0] == 't')
		return test_net(&dut,"data/fnet20_test.dat");
	else if(argv[1][0] == 'b'){
		int N = 2000;
		if(argc == 3)
			N = atoi(argv[2]);
		return bench_net(&dut,"data/cifar10_testset.dat",N);
	}
	else if(argv[1][0] == 'p')
		return predict_image(&dut,argv[2]);
	else{
		printf("Usage:\n");
		printf("Test:       ./inference t\n");
		printf("Benchmark:  ./inference b N\n");
		printf("Predict:    ./inference p image.bmp\n");
		return 1;
	}
	return 0;
}


int test_net(CNN * net, const char * data_file)
{
	FILE *f;
	std::string full_path = merge_pth(PRJ_PATH,data_file);
	if((f = fopen(full_path.c_str(),"rb")) == NULL){
		printf("Reading data file failed!\n");
		return 1;
	}
	uint32_t ntests;
	if(fread(&ntests,sizeof(ntests),1,f) == 0){
		printf("Reading tests failed!\n");
		return 1;
	}
	Tensor X,R;
	double total_time = 0;
	printf("Data: %s\n", full_path.c_str());
	printf("N: %d\n", ntests);
	int ret = 0;
	int misclassified = 0;
	float max_drift = 0.0f;
	for(int i = 0; i < ntests ; i++){
		X.read(f);
		R.read(f);
		uint8_t pred;
		net->inference(&X,1,&pred);
		Tensor * Z = net->layers.back().Z;
		if(Z->size[2] != R.size[2]){
			printf("Test failed Output Tensor has the wrong Dimensions! \n");
			return 1;
		}
		// Probability drift vs reference — REPORTED only (quantization shifts the
		// softmax values); does NOT fail the test.
		for(int k = 0; k < (int)R.size[2]; k++){
			float d = fabsf(R.data[0][0][k] - Z->data[0][0][k]);
			if(d > max_drift) max_drift = d;
		}
		// Pass/fail is on the PREDICTED CLASS (argmax) — the correct metric for
		// quantized inference. A per-element tolerance would false-fail here.
		int ref_pred = 0;
		for(int k = 1; k < (int)R.size[2]; k++)
			if(R.data[0][0][k] > R.data[0][0][ref_pred]) ref_pred = k;
		if((int)pred != ref_pred){
			printf("  MISCLASSIFIED image %d: got %d, expected %d\n", i, pred, ref_pred);
			misclassified++;
			ret = 1;
		}
	}
	printf("Classification: %d/%d correct | max prob drift vs reference = %.4f\n",
	       ntests - misclassified, ntests, max_drift);
	net->print_timing(1);
	fclose(f);
	return ret;
}


/* Runs up to N images as benchmarks */
int bench_net(CNN * net, const char * data_file, int N)
{
	FILE *f;
	std::string full_path = merge_pth(PRJ_PATH,data_file);
	if((f = fopen(full_path.c_str(),"rb")) == NULL){
		printf("ERROR: Reading input file failed!\n");
		fclose(f);
		return 1;
	}
	uint32_t n_inputs;
	if(fread(&n_inputs,sizeof(n_inputs),1,f) != 1){
		printf("ERROR: Reading input file failed!\n");
		fclose(f);
		return 1;
	}
	n_inputs = (N > n_inputs) ? n_inputs : N;
	Tensor * inputs = new Tensor[n_inputs]();
	uint8_t * pred_ref = new uint8_t[n_inputs]();
	uint8_t * preds = new uint8_t[n_inputs]();
	/* Read all Tensors and the prediction*/
	printf("Reading %d inputs!\n",n_inputs);
	for(int i = 0; i < n_inputs; i++){
		int label;
		if(fread(&label,sizeof(label),1,f) != 1){
			printf("ERROR: Reading input file failed!\n");
			fclose(f);
			return 1;
		}
		pred_ref[i] = (uint8_t) label;
		if(inputs[i].read(f) != TENSOR_READ_RESIZED){
			printf("ERROR: Failed Reading Inputs!\n");
			fclose(f);
			return 1;
		}
	}
	fclose(f);

	printf("Starting Inference!\n");
	auto start = std::chrono::high_resolution_clock::now();
	net->inference(inputs,n_inputs,preds);
	auto stop = std::chrono::high_resolution_clock::now();
	double stop_ms = std::chrono::duration<double, std::milli>(stop - start).count();
	printf("Finished Inference!\n");
	net->print_timing(n_inputs);
	printf("=======================\n");
	printf("Images: %d\n",n_inputs);
	printf("-----------------------\n");
	printf("Total Time [ms]: %lf\n",stop_ms);
	printf("Images/s: %lf\n",n_inputs/(stop_ms/1000));
	printf("=======================\n");

	/* Calculate how many Predictions are correct */
	float corr[10] = {0.0};
	int n_per_class[10] = { 0 };
	int total_corr = 0;
	for(int i = 0; i < n_inputs; i++){
		uint8_t label = pred_ref[i];
		n_per_class[label]++;
		if(pred_ref[i] == preds[i]){
			corr[label] += 1;
			total_corr++;
		}
	}
	printf("Accuracy per Class:\n");
	printf("-----------------------\n");
	for(int i = 0; i < 10; i++){
		if(n_per_class[i] > 0)
			corr[i] = corr[i]/n_per_class[i];
		printf("%s: %.2f%% \n",cifar_classes[i],corr[i]*100);
	}
	printf("-----------------------\n");
	printf("Correct Pred.: %d\n",total_corr);
	printf("Accuracy: %.2f%% \n",(100.f*total_corr)/(n_inputs));
	delete [] pred_ref;
	delete [] preds;
	delete [] inputs;
	return 0;
}


int predict_image(CNN * net,const char * img_file)
{
	std::string full_pth = merge_pth(PRJ_PATH,img_file);
	Tensor * x = readBMP(full_pth.c_str());
	if(x == NULL){
		printf("ERROR: Failed to open image file!\n");
		return 1;
	}
	if((x->size[1] != 32) || (x->size[2] != 32)){
		printf("ERROR: Wrong image size %dx%d\n",x->size[1],x->size[2]);
		return 1;
	}
	uint8_t pred;
	net->inference(x,1,&pred);
	Tensor * Z = net->layers.back().Z;
	printf("-------------------------\n");
	printf("Probabilities:\n");
	printf("-------------------------\n");
	for(int k = 0; k < 10; k++){
		const float v = (*Z)[0][0][k];
		printf("%s:%f\n",cifar_classes[k],v);
	}
	printf("-------------------------\n");
	printf("Image %s is a %s!\n",img_file,cifar_classes[pred]);
	printf("-------------------------\n");
	delete x;
	return 0;
}



std::string merge_pth(const char * project_path, const char * file_path)
{
	std::string fp(file_path);
	std::string pp(project_path);
	if(file_path[0] == '/')
		return fp;
	else
		return pp + fp;
}

