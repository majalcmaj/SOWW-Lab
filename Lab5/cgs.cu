/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate graident solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper function CUDA error checking and intialization

const char *sSDKname     = "conjugateGradient";

double mclock(){
	struct timeval tp;

	double sec,usec;
	gettimeofday( &tp, NULL );
	sec    = double( tp.tv_sec );
	usec   = double( tp.tv_usec )/1E6;
	return sec + usec;
}


#define dot_BS     32
#define kernel_BS  32

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
	double RAND_MAXi = 1e6;
	double val_r     = 12.345 * 1e5;

	I[0] = 0, J[0] = 0, J[1] = 1;
	val[0] = (float)val_r/RAND_MAXi + 10.0f;
	val[1] = (float)val_r/RAND_MAXi;
	int start;

	for (int i = 1; i < N; i++)
	{
		if (i > 1)
		{
			I[i] = I[i-1]+3;
		}
		else
		{
			I[1] = 2;
		}

		start = (i-1)*3 + 2;
		J[start] = i - 1;
		J[start+1] = i;

		if (i < N-1)
		{
			J[start+2] = i + 1;
		}

		val[start] = val[start-1];
		val[start+1] = (float)val_r/RAND_MAXi + 10.0f;

		if (i < N-1)
		{
			val[start+2] = (float)val_r/RAND_MAXi;
		}
	}

	I[N] = nz;
}


void cgs_basic(int argc, char **argv, int N, int M){

	//int M = 0, N = 0, 
	int nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-10f;
	const int max_iter = 1000;
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	int k;
	float alpha, beta, alpham1;

	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0)
	{
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
			deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11)
	{
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	/* Generate a random tridiagonal symmetric matrix in CSR format */
	//M = N = 32*64;//10; //1048576;
	printf("M = %d, N = %d\n", M, N);
	nz = (N-2)*3 + 4;
	I = (int *)malloc(sizeof(int)*(N+1));
	J = (int *)malloc(sizeof(int)*nz);
	val = (float *)malloc(sizeof(float)*nz);
	genTridiag(I, J, val, N, nz);

	/*
	   for (int i = 0; i < nz; i++){
	   printf("%d\t", J[i]);
	   }
	   printf("\n");
	   for (int i = 0; i < nz; i++){
	   printf("%2f\t", val[i]);
	   }
	 */

	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);

	for (int i = 0; i < N; i++)
	{
		rhs[i] = 1.0;
		x[i] = 0.0;
	}

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;


	double t_start = mclock();
	cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

	cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);                                // PODMIEN FUNCKJE (I)
	cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        // PODMIEN FUNCKJE (II)

	k = 1;

	while (r1 > tol*tol && k <= max_iter)
	{
		if (k > 1)
		{
			b = r1 / r0;
			cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);                        // PODMIEN FUNCKJE (I)
			cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);            // PODMIEN FUNCKJE (I)
		}
		else
		{
			cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);                    // PODMIEN FUNCKJE (I)
		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax); // PODMIEN FUNCKJE (III)
		cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);                  // PODMIEN FUNCKJE (II)
		a = r1 / dot;

		cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);                    // PODMIEN FUNCKJE (I)
		na = -a;
		cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);                  // PODMIEN FUNCKJE (I)

		r0 = r1;
		cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                    // PODMIEN FUNCKJE (II)
		cudaThreadSynchronize();
		printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		k++;
	}
	printf("TIME OF CGS_BASIC = %f\n", mclock() - t_start);

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i+1]; j++)
		{
			rsum += val[j]*x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);

	cudaDeviceReset();

	printf("Test Summary:  Error amount = %e\n", err);
	//exit((k <= max_iter) ? 0 : 1);


}

	__global__ void
vectorCopy(int elementsCount, const float *src, float *dest)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < elementsCount)
	{
		dest[i] = src[i];
	}
}

	__global__ void
vectorAxpy(int elementsCount, const float *src, float *dest, float alpha)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < elementsCount)
	{
		dest[i] += src[i] * alpha;
	}
}
	__global__ void
vectorScale(int elementsCount, float *vec, float alpha)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < elementsCount)
	{
		vec[i] *= alpha;
	}
}

	__global__ void
sparseMatrixMultiplyByVec(int matrixSize, int nonZeroNumber, float* values, int* rowptr, int* colind, float* x, float* y)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < matrixSize)
	{
		float sub = 0.0f;
		for(int j = rowptr[i] ; j < rowptr[i+1] ; j++) 
		{
			sub += values[j] * x[colind[j]];
		}
		y[i] = sub;
	}
}

	__global__ void
dotProduct(int vectorLength, float *vec1, float* vec2, float* result)
{
	const int TMP_SIZE = 256;
	__shared__ float tmp[TMP_SIZE];

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < vectorLength)
	{
		tmp[threadIdx.x] = vec1[i] * vec2[i];
	}
	else 
	{
		tmp[threadIdx.x] = 0;
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		for(int i = 1 ; i < TMP_SIZE ; i++) {
			tmp [0] += tmp[i];
		}
		atomicAdd(result, tmp[0]);
	}
}


void cgs_TODO(int argc, char **argv, int N, int M){

	//int M = 0, N = 0, 
	int nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-10f;
	const int max_iter = 1000;
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	float* r1d;
	int k;
	float alpha, beta, alpham1;
	int threadsPerBlock = 256;
	int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;


	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0)
	{
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
			deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11)
	{
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	/* Generate a random tridiagonal symmetric matrix in CSR format */
	//M = N = 32*64;//10; //1048576;
	printf("M = %d, N = %d\n", M, N);
	nz = (N-2)*3 + 4;
	I = (int *)malloc(sizeof(int)*(N+1));
	J = (int *)malloc(sizeof(int)*nz);
	val = (float *)malloc(sizeof(float)*nz);
	genTridiag(I, J, val, N, nz);

	/*
	   for (int i = 0; i < nz; i++){
	   printf("%d\t", J[i]);
	   }
	   printf("\n");
	   for (int i = 0; i < nz; i++){
	   printf("%2f\t", val[i]);
	   }
	 */

	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);

	for (int i = 0; i < N; i++)
	{
		rhs[i] = 1.0;
		x[i] = 0.0;
	}

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

	alpha = 1.0;
	alpham1 = -1.0;
	// beta = 0.0;
	r0 = 0.;


	// sparse matrix vector product: d_Ax = A * d_x
	// cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);  // PODMIEN FUNCKJE (ZADANIE-I)
	sparseMatrixMultiplyByVec<<<blocksPerGrid, threadsPerBlock>>>(N, nz, d_val, d_row, d_col, d_x, d_Ax);  // PODMIEN FUNCKJE (ZADANIE-I)


	//azpy: d_r = d_r + alpham1 * d_Ax
	//cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);        			    // PODMIEN FUNCKJE (ZADANIE-I)
	vectorAxpy<<<blocksPerGrid, threadsPerBlock>>>(N, d_Ax, d_r, alpham1);
	//dot:  r1 = d_r * d_r
	// cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        // PODMIEN FUNCKJE (ZADANIE-III)
		
	cudaMalloc(&r1d, sizeof(float));
	r1 = 0;
	cudaMemcpy(r1d, &r1, sizeof(float), cudaMemcpyHostToDevice);
	dotProduct<<<blocksPerGrid, threadsPerBlock>>>(N, d_r, d_r, r1d);                        // PODMIEN FUNCKJE (ZADANIE-III)	
	cudaMemcpy(&r1, r1d, sizeof(float), cudaMemcpyDeviceToHost);
	printf("r1: %f\n", r1);
	k = 1;

	while (r1 > tol*tol && k <= max_iter)
	{
		if (k > 1)
		{
			b = r1 / r0;
			//scal: d_p = b * d_p
			//cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);                        // PODMIEN FUNCKJE (ZADANIE-I)
			vectorScale<<<blocksPerGrid, threadsPerBlock>>>(N, d_p, b);                        // PODMIEN FUNCKJE (ZADANIE-I)
			
			//axpy:  d_p = d_p + alpha * d_r
			// cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);            // PODMIEN FUNCKJE (ZADANIE-I)
			vectorAxpy<<<blocksPerGrid, threadsPerBlock>>>(N, d_r, d_p, alpha);            // PODMIEN FUNCKJE (ZADANIE-I)
		}
		else
		{
			//cpy: d_p = d_r
			vectorCopy<<<blocksPerGrid, threadsPerBlock >>>(N, d_r, d_p);                    				     // PODMIEN FUNCKJE (ZADANIE-I)
		}

		//sparse matrix-vector product: d_Ax = A * d_p
		// cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax); // PODMIEN FUNCKJE (ZADANIE-II)
		sparseMatrixMultiplyByVec<<<blocksPerGrid, threadsPerBlock>>>(N, nz, d_val, d_row, d_col, d_p, d_Ax); // PODMIEN FUNCKJE (ZADANIE-II)
		
		// cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);                  // PODMIEN FUNCKJE (ZADANIE-III)
		dot = 0.0;
		cudaMemcpy(r1d, &dot, sizeof(float), cudaMemcpyHostToDevice);
		dotProduct<<<blocksPerGrid, threadsPerBlock>>>(N, d_p, d_Ax, r1d);                        // PODMIEN FUNCKJE (ZADANIE-III)	
		cudaMemcpy(&dot, r1d, sizeof(float), cudaMemcpyDeviceToHost);
		
		a = r1 / dot;

		//axpy: d_x = d_x + a*d_p
		cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);                    // PODMIEN FUNCKJE (ZADANIE-I)
		na = -a;

		//axpy:  d_r = d_r + na * d_Ax
		cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);                  // PODMIEN FUNCKJE (ZADANIE-I)

		r0 = r1;

		//dot: r1 = d_r * d_r
		// cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                    // PODMIEN FUNCKJE (ZADANIE-III)
		
		r1 = 0.0;
		cudaMemcpy(r1d, &r1, sizeof(float), cudaMemcpyHostToDevice);
		dotProduct<<<blocksPerGrid, threadsPerBlock>>>(N, d_r, d_r, r1d);                        // PODMIEN FUNCKJE (ZADANIE-III)	
		cudaMemcpy(&r1, r1d, sizeof(float), cudaMemcpyDeviceToHost);

		
		cudaThreadSynchronize();
		printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		k++;
	}

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i+1]; j++)
		{
			rsum += val[j]*x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);

	cudaDeviceReset();

	printf("Test Summary:  Error amount = %e\n", err);
	//exit((k <= max_iter) ? 0 : 1);


}







int main(int argc, char **argv)
{
	//int N = 1e6;//1 << 20;
	//int N = 256 * (1<<10)  -10 ; //1e6;//1 << 20;
	int N = 1e5;
	int M = N; 

	cgs_basic(argc, argv, N, M);

	cgs_TODO(argc, argv, N, M);
}
