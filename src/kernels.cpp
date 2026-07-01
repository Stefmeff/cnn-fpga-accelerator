#include "kernels.h"
/* 
 * Applies a 2d convolution on a 3D X using W: Z= W (conv) X + b
 * Stride=1, kernel size = (W->size[2])
 * Tensor * X:		Input Tensor
 * Tensor * W:		Array of N weight Tensors (N == Z.size[0]) 
 * Tensor * Z:		Output Tensor 
 * Tensor * b:		Bias 
 */
void conv2d(Tensor * X, Tensor * W ,  Tensor * B, Tensor * Z)
{
	// No padding required!
	// Get boundaries of loops
	int ksize = W->size[2];
	int w_start = -ksize/2;
	int w_stop = ksize/2 + ksize%2;
	int z_start = ksize/2;
	int z_stop= X->size[2] - ksize/2 +(1-ksize%2);
	// Loop through output channels
	int i,j,k,l,p,q;
	for(i = 0; i < Z->size[0]; i++){
		FLOAT ** z = (*Z)[i];
		// Set outputmatrix to values of b
		for(j = 0; j < Z->size[1]; j++){
			for(k = 0; k < Z->size[2] ; k++){
				z[j][k] = (*B)[0][0][i];
			}
		}
		// Loop through the input channels
		for(j = 0; j < X->size[0] ; j++){
			FLOAT ** w = (W[i])[j];
			FLOAT ** x = (*X)[j];
			// Convolution!
			for(k = z_start; k < z_stop; k++){
				for(l = z_start; l < z_stop; l++){
					float * sum = &(z[k - z_start][l - z_start]);
					for(p = w_start; p < w_stop ; p++){
						for(q = w_start; q < w_stop ; q++){
							*sum += w[p - w_start][q - w_start]
								*x[k + p][l + q];
						}
					}
				}
			}
		}
	}
}


/*
 * Applies a max pool layer on X (size=2, stride = 2)
 * Tensor * X:	input Tensor
 * Tensor * Z:	output Tensor
 */
void maxPool(Tensor * X, Tensor * Z)
{
	for(int i = 0; i < X->size[0]; i++){
		FLOAT ** x = (*X)[i];
		FLOAT ** z = (*Z)[i];
		for(int j = 0; j < Z->size[1];j++){
			for(int k = 0; k < Z->size[2];k++){
				z[j][k] = 0;
				for(int p = j*2; p < (j*2 + 2); p++){
					for(int q = k*2; q < (k*2 + 2); q++){
						if(x[p][q] > z[j][k])
							z[j][k] = x[p][q];
					}
				}
			}
		}
	}
}

/*
 * Applies an avg pool on X (Every channel gets fully pooled)
 * Output dim Z: X[0],1,1
 * Tensor * X:	input Tensor
 * Tensor * Z:	output Tensor
 */
void avgPool(Tensor * X, Tensor * Z)
{
	for(int i = 0; i < Z->size[0]; i++){
		FLOAT * z = (*Z)[0][0];
		z[i] = 0;
		FLOAT * x = (*X)[i][0];
		for(int j = 0; j < (X->size[1]*X->size[2]);j++){
			z[i] += x[j];
		}
		z[i] = z[i]/(X->size[1]*X->size[2]);
	}
}


/*
 * Applies a Linear layer: z = Wx + b 
 * Flatten the input if required 
 * Tensor *	X: input Tensor
 * Tensor *	W: weight Matrix (in Tensor form)
 * Tensor *	B: bias array (in Tensor form)
 * Tensor *	Z: output array (in Tensor form)
 */
void Linear(Tensor * X, Tensor * W, Tensor * B, Tensor * Z)
{
	int K = X->size[2] * X->size[1];
	int N = W->size[1];
	// Outputs are stored inside the x-dimesion of the Z tensor 
	// So is the bias
	FLOAT * z = (*Z)[0][0];
	FLOAT * b = (*B)[0][0];
	// Weigth is just the first W matrix
	FLOAT ** w = (*W)[0];
	int i,j,k;
	// Loop through output neurons
	for(i = 0; i < N; i++){
		float sum = b[i];
		for(j = 0; j < X->size[0]; j++){
			FLOAT * x = (*X)[j][0];
			for(k = 0 ; k < K; k++){
				sum += w[i][j*K + k]*x[k];
			}
		}
		z[i] = sum;
	}
}



/*
 * Applies the ReLU activation function: Z = ReLU(X)
 * Tensor * X: input Tensor
 * Tensor * Z: output Tensor
 */
void ReLU(Tensor * X , Tensor * Z)
{
	int N = X->size[0]*X->size[1]*X->size[2];
	FLOAT * z = (*Z)[0][0];
	FLOAT * x = (*X)[0][0];
	for(int i = 0; i < N; i++){
		z[i] = (x[i] > 0) ? x[i] : 0;
	}
}


/*
 * Applies the Softmax activation function z = exp(x_i)/sum(exp(x_j))
 * This is a stable Softmax implementation
 * Tensor * X: input vector in Tensor form
 * Tensor * Z: output vector in Tensor form
 */
void Softmax(Tensor * X, Tensor * Z)
{
	FLOAT * x = (*X)[0][0];
	FLOAT * z = (*Z)[0][0];
	int N = X->size[2];
	// Get the max value of a
	float maxx = 0;
	for(int i = 0; i < N ; i++){
		if(x[i] > maxx)
			maxx = x[i];
	}
	float sumexp = 0;
	for(int i =0; i< N;i++){
		z[i] = exp(x[i] - maxx);
		sumexp += z[i];
	}
	for(int i =0; i < N ; i++){
		z[i] = z[i]/sumexp;
	}
}
