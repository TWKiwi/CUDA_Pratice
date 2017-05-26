#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ostream>
#include <thread>
#include <fstream>
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;


#define DATA_SIZE 1048576
#define BLOCK_NUM 32
#define THREAD_NUM 256
#define MODE 0

std::string CMD = "";
int data[DATA_SIZE];
float f_data[DATA_SIZE];
float f_c_data[DATA_SIZE];
float f_fac_data[DATA_SIZE];
bool InitCUDA();
void GenerateNumbers(int *number, int size);
void matgen(float* a, int lda, int n);
void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n);
void ArrayCompute();
void ArrayCompute_multiple_threads();
void ArrayCompute_multiple_threads_continuous_access();
void ArrayCompute_multiple_threads_blocks_continuous_access();
void ArrayCompute_shared_multiple_threads_blocks_continuous_access();
void ArrayCompute_shared_multiple_threads_blocks_continuous_access_treesum();
void ArrayCompute_shared_multiple_threads_blocks_continuous_access_better_treesum();

__global__ static void sumOfSquares(int *num, int* result);
__global__ static void sumOfSquares_multiple_threads(int *num, int* result);
__global__ static void sumOfSquares_multiple_threads_continuous_access(int *num, int* result);
__global__ static void sumOfSquares_multiple_threads_blocks_continuous_access(int *num, int* result);
__global__ static void sumOfSquares_shared_multiple_threads_blocks_continuous_access(int *num, int* result);
__global__ static void sumOfSquares_shared_multiple_threads_blocks_continuous_access_treesum(int *num, int* result);
__global__ static void sumOfSquares_shared_multiple_threads_blocks_continuous_access_better_treesum(int *num, int* result);

void FloatArrayMultiCompute(int n);
clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n);
__global__ static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n);

void FloatArrayMultiCompute_KSF(int n);
clock_t matmultCUDA_KSF(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n);
__global__ static void matMultCUDA_KSF(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n);

void FloatArrayMultiCompute_KSF_shared_pitch(int n);
clock_t matmultCUDA_KSF_shared_pitch(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n);
__global__ static void matMultCUDA_KSF_shared_pitch(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n);

void inverse_matrix();
float determinant(float *data, int k);
void cofactor(int f, float *inverse);
void transpose(int r, float *inverse);

void inverse_matrix();
void inverse_matrix_CPU(int k);
void inverse_matrix_GPU(const int n);




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  ���C���K              //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cudaDeviceProp prop;

void func_cmd()
{
	std::string cmd = CMD;
	system(cmd.c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  �D�i�J�I              //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{
	if (MODE == 0)
	{

		if (InitCUDA())
		{
			ArrayCompute();
			printf("(1).%dThreads��}����\n", THREAD_NUM);
			ArrayCompute_multiple_threads();
			printf("(2)(3).%dThreads �s��O����s������\n", THREAD_NUM);
			ArrayCompute_multiple_threads_continuous_access();
			printf("(4).%dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_multiple_threads_blocks_continuous_access();
			printf("(5).Shared Memory %dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_shared_multiple_threads_blocks_continuous_access();
			printf("(6).TreeSum alg. Shared Memory %dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_shared_multiple_threads_blocks_continuous_access_treesum();
			printf("(7).��}TreeSum alg. Shared Memory %dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_shared_multiple_threads_blocks_continuous_access_better_treesum();
			printf("//////////////FLOAT////////////////\n");
			printf("/////////�B�I�ƪ��x�}���k//////////\n");
			int n = 0;
			printf("��J�x�}�j�p(�@�ӥ����):");
			scanf_s(" %d", &n);
			printf("(1).�@�목��\n");
			FloatArrayMultiCompute(n);
			printf("(2).Kahan'sSummation Formula��}\n");
			FloatArrayMultiCompute_KSF(n);
			printf("(3).KSF��} Shared Memory Pitch\n");
			FloatArrayMultiCompute_KSF_shared_pitch(n);
			printf("(4).GPU�ϯx�}\n");
			inverse_matrix();


		}

		printf("\nDone!");
		system("pause");
		return 0;
	}
	else
	{
		/*
		 * ���ư�����h�Ұ�exe�ĪG�񪽱���system()�n�A���p�O�]�������ΥD������s��Ұ����ε{�������Y�A�t���q�L���F�C
		 * �o�O�ڪ��q���C
		 */
		CMD = "cd ..\\Multi_Window_Display && start CUDA.exe && start CUDA_m_t.exe && start CUDA_m_t_c_a_(256).exe && start CUDA_m_t_c_a_(512).exe && start CUDA_m_t_b_c_a.exe && start CUDA_s_m_t_b_c_a.exe && start CUDA_s_m_t_b_c_a_t.exe && start CUDA_s_m_t_b_c_a_b_t.exe";
		std::thread thread(func_cmd);
		thread.join();

		return 0;
	}




}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  �򥻸�T              //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "No Device!!");
		return false;
	}

	int i;
	for (i = 0; i < count; i++)
	{

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				printf("��%d�i��d�W�� -> %s\n", count, prop.name);
				printf("--CUDA���� -> %d.%d\n", prop.major, prop.minor);
				char msg[256];
				sprintf_s(msg, "--�`Global Memory -> %.0f MBytes (%llu bytes)\n",
					(float)prop.totalGlobalMem / 1048576.0f, (unsigned long long) prop.totalGlobalMem);
				printf("%s", msg);
				printf("--%2d �� Multiprocessors\n", prop.multiProcessorCount);
				printf("--�C��Multiprocessors�̳̤j������ƶq:%d\n", prop.maxThreadsPerMultiProcessor);
				printf("--�C��Block�̳̤j������ƶq:%d\n", prop.maxThreadsPerBlock);
				printf("--GPU �̤j�ɯ�: %.0f MHz (%0.2f GHz)\n", prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
				printf("--�̤j�O����ɯ�: %.0f Mhz (%0.2f GHz)\n", prop.memoryClockRate * 1e-3f, prop.memoryClockRate * 1e-6f);
				break;
			}
		}
	}

	if (i == count)
	{
		fprintf(stderr, "No device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  �����                //////////////////////////////////////////
/////////////////////////////////////                  ����M                //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * �ͶüƦ�C
 */
void GenerateNumbers(int *number, int size)
{
	for (int i = 0; i < size; i++)
	{
		number[i] = rand() % 10;
	}
}

/*
 * 1 Thread����
 */
void ArrayCompute()
{
	float timeValue;


	//-----------------------------------------------

	GenerateNumbers(data, DATA_SIZE);
	int* gpudata, *result;
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares << <1, 1, 0 >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum;
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);

	printf("--sum (GPU): %d\n", sum);
	printf("--����ɶ� (GPU): %f\n", float(timeValue) / CLOCKS_PER_SEC);
	//-----------------------------------------------

	sum = 0;
	clock_t cpu_time = clock();
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += data[i] * data[i];
	}
	printf("--sum (CPU): %d\n", sum);
	printf("--����ɶ� (CPU): %f\n", float(clock() - cpu_time) / CLOCKS_PER_SEC);

}
/*
* �쪩����[�`�{��
*/
__global__ static void sumOfSquares(int *num, int* result)
{
	int sum = 0;
	int i;
	for (i = 0; i < DATA_SIZE; i++) {
		sum += num[i] * num[i];
	}
	*result = sum;
}
/*
 * Multiple Threads����
 */
void ArrayCompute_multiple_threads()
{
	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;

	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares_multiple_threads << <1, THREAD_NUM, 0 >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum[THREAD_NUM];
	cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	//-----------------------------------------------
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM; i++) {
		final_sum += sum[i];
	}
	float clock_cycle = prop.clockRate * 1e-3f * (float(timeValue) / CLOCKS_PER_SEC);
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // �u�A�Ω�32�줸��ƫe�� (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--����ɶ� (GPU): %f | �ɯ�: %fMHz | �O�����W�e:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* ��}�ᥭ��[�`�{��
* multiple threads blocks
*/
__global__ static void sumOfSquares_multiple_threads(int *num, int* result)
{
	const int tid = threadIdx.x;
	const int size = DATA_SIZE / THREAD_NUM;
	int sum = 0;
	int i;
	for (i = tid * size; i < (tid + 1) * size; i++) {
		sum += num[i] * num[i];
	}
	result[tid] = sum;
}
/*
* Multiple Threads����
* �s��O����s������
*/
void ArrayCompute_multiple_threads_continuous_access()
{
	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;

	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares_multiple_threads_continuous_access << <1, THREAD_NUM, 0 >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum[THREAD_NUM];
	cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	//-----------------------------------------------
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM; i++) {
		final_sum += sum[i];
	}
	float clock_cycle = prop.clockRate * 1e-3f * (float(timeValue) / CLOCKS_PER_SEC);
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // �u�A�Ω�32�줸��ƫe�� (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--����ɶ� (GPU): %f | �ɯ�: %fMHz | �O�����W�e:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* ��}�ᥭ��[�`�{��
* multiple threads �s��O����s��
*/
__global__ static void sumOfSquares_multiple_threads_continuous_access(int *num, int* result)
{
	const int tid = threadIdx.x;
	int sum = 0;
	int i;
	for (i = tid; i < DATA_SIZE; i += THREAD_NUM) {
		sum += num[i] * num[i];
	}
	result[tid] = sum;
}
/*
* Multiple Threads Blocks����
* �s��O����s������
*/
void ArrayCompute_multiple_threads_blocks_continuous_access()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;

	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares_multiple_threads_blocks_continuous_access << <BLOCK_NUM, THREAD_NUM, 0 >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum[THREAD_NUM * BLOCK_NUM];
	cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);
	//-----------------------------------------------
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM * BLOCK_NUM; i++) {
		final_sum += sum[i];
	}

	float clock_cycle = prop.clockRate * 1e-3f * (float(timeValue) / CLOCKS_PER_SEC);
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // �u�A�Ω�32�줸��ƫe�� (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--����ɶ� (GPU): %f | �ɯ�: %fMHz | �O�����W�e:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* ��}�ᥭ��[�`�{��
* multiple threads blocks
*/
__global__ static void sumOfSquares_multiple_threads_blocks_continuous_access(int *num, int* result)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int sum = 0;
	int i;
	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
		sum += num[i] * num[i];
	}
	result[bid * THREAD_NUM + tid] = sum;
}
/*
* Shared Multiple Threads Blocks����
* �s��O����s������
*/
void ArrayCompute_shared_multiple_threads_blocks_continuous_access()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;

	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares_shared_multiple_threads_blocks_continuous_access << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum[THREAD_NUM * BLOCK_NUM];
	cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);
	//-----------------------------------------------
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM * BLOCK_NUM; i++) {
		final_sum += sum[i];
	}

	float clock_cycle = prop.clockRate * 1e-3f * (float(timeValue) / CLOCKS_PER_SEC);
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // �u�A�Ω�32�줸��ƫe�� (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--����ɶ� (GPU): %f | �ɯ�: %fMHz | �O�����W�e:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* ��}�ᥭ��[�`�{��
* shared multiple threads blocks
*/
__global__ static void sumOfSquares_shared_multiple_threads_blocks_continuous_access(int *num, int* result)
{
	extern __shared__ int shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	int i;
	shared[tid] = 0;

	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE;
		i += BLOCK_NUM * THREAD_NUM) {
		shared[tid] += num[i] * num[i];
	}
	__syncthreads();

	if (tid == 0) {
		for (i = 1; i < THREAD_NUM; i++) {
			shared[0] += shared[i];
		}
		result[bid] = shared[0];
	}
}
/*
* Shared Multiple Threads Blocks����
* �s��O����s������
* TreeSum alg.
*/
void ArrayCompute_shared_multiple_threads_blocks_continuous_access_treesum()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;

	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares_shared_multiple_threads_blocks_continuous_access_treesum << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum[THREAD_NUM * BLOCK_NUM];
	cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);
	//-----------------------------------------------
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM * BLOCK_NUM; i++) {
		final_sum += sum[i];
	}

	float clock_cycle = prop.clockRate * 1e-3f * (float(timeValue) / CLOCKS_PER_SEC);
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // �u�A�Ω�32�줸��ƫe�� (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--����ɶ� (GPU): %f | �ɯ�: %fMHz | �O�����W�e:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* ��}�ᥭ��[�`�{��
* shared multiple threads blocks
* TreeSum alg.
*/
__global__ static void sumOfSquares_shared_multiple_threads_blocks_continuous_access_treesum(int *num, int* result)
{
	extern __shared__ int shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int i;
	int offset = 1, mask = 1;
	shared[tid] = 0;
	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
		shared[tid] += num[i] * num[i];
	}
	__syncthreads();
	while (offset < THREAD_NUM) {
		if ((tid & mask) == 0) {
			shared[tid] += shared[tid + offset];
		}
		offset += offset;
		mask = offset + mask;
		__syncthreads();
	}
	if (tid == 0) {
		result[bid] = shared[0];
	}
}
/*
* Shared Multiple Threads Blocks����
* �s��O����s������
* ��}TreeSum alg.
*/
void ArrayCompute_shared_multiple_threads_blocks_continuous_access_better_treesum()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;

	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	sumOfSquares_shared_multiple_threads_blocks_continuous_access_better_treesum << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	int sum[THREAD_NUM * BLOCK_NUM];
	cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);
	//-----------------------------------------------
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM * BLOCK_NUM; i++) {
		final_sum += sum[i];
	}

	float clock_cycle = prop.clockRate * 1e-3f * (float(timeValue) / CLOCKS_PER_SEC);
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // �u�A�Ω�32�줸��ƫe�� (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--����ɶ� (GPU): %f | �ɯ�: %fMHz | �O�����W�e:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);

}
/*
* ��}�ᥭ��[�`�{��
* shared multiple threads blocks
* ��}TreeSum alg.
*/
__global__ static void sumOfSquares_shared_multiple_threads_blocks_continuous_access_better_treesum(int *num, int* result)
{
	extern __shared__ int shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int i;
	int offset = 1, mask = 1;
	shared[tid] = 0;
	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
		shared[tid] += num[i] * num[i];
	}
	__syncthreads();

	if (tid < 128) { shared[tid] += shared[tid + 128]; }
	__syncthreads();
	if (tid < 64) { shared[tid] += shared[tid + 64]; }
	__syncthreads();
	if (tid < 32) { shared[tid] += shared[tid + 32]; }
	__syncthreads();
	if (tid < 16) { shared[tid] += shared[tid + 16]; }
	__syncthreads();
	if (tid < 8) { shared[tid] += shared[tid + 8]; }
	__syncthreads();
	if (tid < 4) { shared[tid] += shared[tid + 4]; }
	__syncthreads();
	if (tid < 2) { shared[tid] += shared[tid + 2]; }
	__syncthreads();
	if (tid < 1) { shared[tid] += shared[tid + 1]; }
	__syncthreads();

	if (tid == 0) {
		result[bid] = shared[0];
	}

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  �B�I��                //////////////////////////////////////////
/////////////////////////////////////                 �x�}�ۭ�               //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
* �ͦ��üƯB�I�Ưx�}
*/
void matgen(float* a, int lda, int n)
{
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a[i * lda + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
		}
	}
}

/*
* CPU���x�}���k
*/
void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			double t = 0;
			for (k = 0; k < n; k++) {
				t += a[i * lda + k] * b[k * ldb + j];
			}
			c[i * ldc + j] = t;
		}
	}
}

/*
* CPU���үx�}�ۭ����G
* �L�X�̤j�~�t�ȡA�����~�t��
*/
void compare_mat(const float* a, int lda, const float* b, int ldb, int n)
{
	float max_err = 0;
	float average_err = 0;
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (b[i * ldb + j] != 0) {
				float err = fabs((a[i * lda + j] - b[i * ldb + j]) / b[i * ldb + j]);
				if (max_err < err) max_err = err; average_err += err;
			}
		}
	}
	printf("�̤j�~�t��: %g �����~�t��: %g\n", max_err, average_err / (n * n));

}

/*
*�B�I�Ưx�}���k
*/
void FloatArrayMultiCompute(int n)
{
	float *a, *b, *c, *d;
	float cpu_time_use_per_sec = 0.0;
	a = (float*)malloc(sizeof(float)* n * n);
	b = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	d = (float*)malloc(sizeof(float)* n * n);
	srand(0);
	matgen(a, n, n);
	matgen(b, n, n);
	clock_t time = matmultCUDA(a, n, b, n, c, n, n);
	clock_t cpu_time_start = clock();//CPU�p�ɶ}�l
	matmult(a, n, b, n, d, n, n);
	cpu_time_use_per_sec = (float)(clock() - cpu_time_start) / CLOCKS_PER_SEC;//CPU�p�ɵ����õ���
	compare_mat(c, n, d, n, n);
	double sec = (double)time / CLOCKS_PER_SEC;
	printf("CPU Time used: %f\n", cpu_time_use_per_sec);
	printf("GPU Time used: %f (%f GFLOPS)\n", sec, 2.0 * n * n * n / (sec * 1E9));
}

/*
 * ���I���ө��լ�����o��Ѯv�n�S�a��cudaMalloc�BcudaMemcpy2D�BcudaFree�Ҽ{�i�h�p��ɶ��A���e�N���Ҽ{...
 */
clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	float timeValue;
	float *ac, *bc, *cc;

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	cudaMalloc((void**)&ac, sizeof(float)* n * n);
	cudaMalloc((void**)&bc, sizeof(float)* n * n);
	cudaMalloc((void**)&cc, sizeof(float)* n * n);
	cudaMemcpy2D(ac, sizeof(float)* n, a, sizeof(float)* lda, sizeof(float)* n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, sizeof(float)* n, b, sizeof(float)* ldb, sizeof(float)* n, n, cudaMemcpyHostToDevice);

	int blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
	matMultCUDA << <blocks * n, THREAD_NUM >> >(ac, n, bc, n, cc, n, n);
	cudaMemcpy2D(c, sizeof(float)* ldc, cc, sizeof(float)* n, sizeof(float)* n, n, cudaMemcpyDeviceToHost);
	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);


	return timeValue;
}

/*
 * �x�}�ۭ���Kernel�禡
 */
__global__ static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int idx = bid * blockDim.x + tid;
	const int row = idx / n;
	const int column = idx % n;
	int i;
	if (row < n && column < n) {
		float t = 0;
		for (i = 0; i < n; i++) {
			t += a[row * lda + i] * b[i * ldb + column];
		}
		c[row * ldc + column] = t;
	}
}

/*
 * �B�I�Ưx�}���k
 * Kahan'sSummation Formula
 */
void FloatArrayMultiCompute_KSF(int n)
{
	float *a, *b, *c, *d;
	a = (float*)malloc(sizeof(float)* n * n);
	b = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	d = (float*)malloc(sizeof(float)* n * n);
	srand(0);
	matgen(a, n, n);
	matgen(b, n, n);
	clock_t time = matmultCUDA_KSF(a, n, b, n, c, n, n);
	matmult(a, n, b, n, d, n, n);
	compare_mat(c, n, d, n, n);
	double sec = (double)time / CLOCKS_PER_SEC;
	printf("Time used: %f (%f GFLOPS)\n", sec, 2.0 * n * n * n / (sec * 1E9));
}

/*
* ���I���ө��լ�����o��Ѯv�n�S�a��cudaMalloc�BcudaMemcpy2D�BcudaFree�Ҽ{�i�h�p��ɶ��A���e�N���Ҽ{...
* Kahan'sSummation Formula
*/
clock_t matmultCUDA_KSF(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	float timeValue;
	float *ac, *bc, *cc;

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	cudaMalloc((void**)&ac, sizeof(float)* n * n);
	cudaMalloc((void**)&bc, sizeof(float)* n * n);
	cudaMalloc((void**)&cc, sizeof(float)* n * n);
	cudaMemcpy2D(ac, sizeof(float)* n, a, sizeof(float)* lda, sizeof(float)* n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, sizeof(float)* n, b, sizeof(float)* ldb, sizeof(float)* n, n, cudaMemcpyHostToDevice);

	int blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
	matMultCUDA_KSF << <blocks * n, THREAD_NUM >> >(ac, n, bc, n, cc, n, n);
	cudaMemcpy2D(c, sizeof(float)* ldc, cc, sizeof(float)* n, sizeof(float)* n, n, cudaMemcpyDeviceToHost);
	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);


	return timeValue;
}

/*
* �x�}�ۭ���Kernel�禡
* Kahan'sSummation Formula
*/
__global__ static void matMultCUDA_KSF(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int idx = bid * blockDim.x + tid;
	const int row = idx / n;
	const int column = idx % n;
	int i;
	if (row < n && column < n) {
		float t = 0;
		float y = 0;
		for (i = 0; i < n; i++) {
			//�o��u���ݤ����b�F��...
			float r;
			y -= a[row * lda + i] * b[i * ldb + column];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		c[row * ldc + column] = t; //�o��O�ڥ[�W�h���A�]�����[�o�����~�t�ȳ��|�O1�A�D�n���D�I�٬O�W��...
	}
	//�o�q�g�k����n�z�ѡA�i�O��Ѯv�o�y�L���P...
	/*if (row < n && column < n) {
		float sum = 0;
		float z = 0;
		for (i = 0; i < n; i++) {
		float y = a[row * lda + i] * b[i * ldb + column] - z;
		float t = sum + y;
		z = (t - sum) - y;
		sum += t;
		}
		c[row * ldc + column] = sum;
		}*/
}


/*
* �B�I�Ưx�}���k
* Kahan'sSummation Formula
* Shared memory
* Pitch
*/
void FloatArrayMultiCompute_KSF_shared_pitch(int n)
{
	float *a, *b, *c, *d;
	a = (float*)malloc(sizeof(float)* n * n);
	b = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	d = (float*)malloc(sizeof(float)* n * n);
	srand(0);
	matgen(a, n, n);
	matgen(b, n, n);
	clock_t time = matmultCUDA_KSF_shared_pitch(a, n, b, n, c, n, n);
	matmult(a, n, b, n, d, n, n);
	compare_mat(c, n, d, n, n);
	double sec = (double)time / CLOCKS_PER_SEC;
	printf("Time used: %f (%f GFLOPS)\n", sec, 2.0 * n * n * n / (sec * 1E9));
}

/*
* ���I���ө��լ�����o��Ѯv�n�S�a��cudaMalloc�BcudaMemcpy2D�BcudaFree�Ҽ{�i�h�p��ɶ��A���e�N���Ҽ{...
* Kahan'sSummation Formula
* Shared memory
* Pitch
*/
clock_t matmultCUDA_KSF_shared_pitch(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	float timeValue;
	float *ac, *bc, *cc;

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	size_t pitch_a, pitch_b, pitch_c;

	cudaMallocPitch((void**)&ac, &pitch_a, sizeof(float)* n, n);
	cudaMallocPitch((void**)&bc, &pitch_b, sizeof(float)* n, n);
	cudaMallocPitch((void**)&cc, &pitch_c, sizeof(float)* n, n);
	cudaMemcpy2D(ac, pitch_a, a, sizeof(float)* lda, sizeof(float)* n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, pitch_b, b, sizeof(float)* ldb, sizeof(float)* n, n, cudaMemcpyHostToDevice);

	int blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
	matMultCUDA_KSF_shared_pitch << <n, THREAD_NUM, sizeof(float)* n >> >(ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);
	cudaMemcpy2D(c, sizeof(float)* ldc, cc, pitch_c, sizeof(float)* n, n, cudaMemcpyDeviceToHost);
	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);


	return timeValue;
}

/*
* �x�}�ۭ���Kernel�禡
* Kahan'sSummation Formula
* Shared memory
* Pitch
*/
__global__ static void matMultCUDA_KSF_shared_pitch(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
{
	extern __shared__ float shared_data[];
	const int tid = threadIdx.x;
	const int row = blockIdx.x;
	int i, j;

	for (i = tid; i < n; i += blockDim.x) {
		shared_data[i] = a[row * lda + i];
	}

	__syncthreads();

	for (j = tid; j < n; j += blockDim.x) {
		float t = 0;
		float y = 0;
		for (i = 0; i < n; i++) {
			float r;
			y -= shared_data[i] * b[i * ldb + j];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		c[row * ldc + j] = t;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  �ϯx�}                //////////////////////////////////////////
/////////////////////////////////////           GPU��ʿ�J���٥�����        //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void inverse_matrix()
{
	int i, j, k;
	char ans;
	printf("\n��J�x�}�j�p : ");
	scanf(" %d", &k);
	/*printf("\n�ȬO�_�üƲ���?(y|n) : ");
	scanf(" %c", &ans);
	if (ans == 'y')
	{
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				f_data[i * k + j] = (int)rand() % 100;
			}
		}
	}
	else if (ans == 'n')
	{
		printf("\n��J�Ȩ� %d x %d �x�} : \n", k, k);
		for (i = 0; i < k; i++)
		{
			for (j = 0; j < k; j++)
			{
				scanf(" %f", &f_data[i * k + j]);
			}
		}
	}
	else
	{
		inverse_matrix();
	}
	printf("��J�x�}��:\n");
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < k; j++)
		{
			printf("\t%f", f_data[i * k + j]);
		}
		printf("\n");
	}*/

	inverse_matrix_GPU(k);
	//inverse_matrix_CPU(k);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////�}�l��@��CPU�ϯx�}///////////////////////////////////////////////////

/*
* �ϯx�}
* http://www.ccodechamp.com/c-program-to-find-inverse-of-matrix/
*/
void inverse_matrix_CPU(int k)
{
	printf("\n//////////////////////////////////////////////////////////////////\n");
	printf("\n/////////////////////////CPU�ϯx�}�}�l////////////////////////////\n");
	float d = 0.0;
	float* inverse = (float*)malloc(sizeof(float)* k * k);;
	clock_t determinant_start = clock();
	d = determinant(f_data, k);

	if (d == 0.0)
	{
		printf("��J�Ȧ��~�A�L�k�D�o�ϯx�}\n");
		printf("��C���� = %f, �@�Ӯ�%f\n", d, (float)(clock() - determinant_start) / CLOCKS_PER_SEC);
	}
	else
	{
		cofactor(k, inverse);
		printf("��C���� = %f, �@�Ӯ�%f\n", d, (float)(clock() - determinant_start) / CLOCKS_PER_SEC);

		printf("\n���Ҥϯx�}: \n");

		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) {
				f_c_data[i*k + j] = 0; /*��l�ư}�CC */
				for (int k_i = 0; k_i < k; k_i++) {
					size_t index = k_i * k + j;
					f_c_data[i*k + j] += f_data[i*k + k_i] * inverse[index]; /*�}�CA���W�}�CB,�s�J�}�CC */
					//printf("\nf_data[%d][%d]:%f inverse[%d]:%f", i, k_i, f_data[i*k + k_i], index, inverse[index]); /*��X�}�CC */
				}
				printf("\t%f", f_c_data[i*k + j]);
			}
			printf("\n");
		}
	}


}

float determinant(float *data, int k)
{
	float s = 1, det = 0;
	float *f_b_data;
	f_b_data = (float*)malloc(sizeof(float)* k * k);
	int i, j, m, n, c;
	if (k == 1)
	{
		return (data[0]);
	}

	det = 0;
	for (c = 0; c < k; c++)
	{
		m = 0;
		n = 0;
		for (i = 0; i < k; i++)
		{
			for (j = 0; j < k; j++)
			{
				f_b_data[i + j*k] = 0;
				if (i != 0 && j != c)
				{
					f_b_data[m*k + n] = data[i*k + j];
					//if (k == 5) printf("f_b_data[%d][%d]:%f = data[%d][%d]:%f;\n", m, n, f_b_data[m*k + n], i, j, data[i*k + j]);
					if (n < (k - 2))
						n++;
					else
					{
						n = 0;
						m++;
					}
				}
			}
		}
		det = det + s * (data[c] * determinant(f_b_data, k - 1));
		//if (k == 2) printf("%f + %f * (data[%d]:%f * %f);\n", det, s,c, data[c], determinant(f_b_data, k - 1));
		s = -1 * s;
	}


	return det;
}



void cofactor(int f, float *inverse)
{
	int p, q, m, n, i, j;
	float *f_b_data;
	f_b_data = (float*)malloc(sizeof(float)* f * f);
	for (q = 0; q < f; q++)
	{
		for (p = 0; p < f; p++)
		{
			m = 0;
			n = 0;
			for (i = 0; i < f; i++)
			{
				for (j = 0; j < f; j++)
				{
					if (i != q && j != p)
					{
						f_b_data[m*f + n] = f_data[i*f + j];
						if (n < (f - 2))
							n++;
						else
						{
							n = 0;
							m++;
						}
					}
				}
			}
			f_fac_data[q*f + p] = pow(-1, q + p) * determinant(f_b_data, f - 1);
		}
	}
	transpose(f, inverse);
}
/*�D�ϯx�}*/
void transpose(int r, float *inverse)
{
	int i, j;
	float *f_b_data;
	f_b_data = (float*)malloc(sizeof(float)* r * r);
	float d;

	for (i = 0; i < r; i++)
	{
		for (j = 0; j < r; j++)
		{
			f_b_data[i*r + j] = f_fac_data[j*r + i];
		}
	}
	d = determinant(f_data, r);
	for (i = 0; i < r; i++)
	{
		for (j = 0; j < r; j++)
		{
			inverse[i*r + j] = f_b_data[i*r + j] / d;
		}
	}
	printf("�ϯx�}�� : \n");

	for (i = 0; i < r; i++)
	{
		for (j = 0; j < r; j++)
		{
			printf("\t%f", inverse[i*r + j]);
		}
		printf("\n");
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////�}�l��@��GPU�ϯx�}///////////////////////////////////////////////////
/*
 * �ק��
 * https://github.com/ZhengzhongSun/Matrix-Inversion-with-CUDA
 */
void matrix_gen(float *L, int dimension){
	int row, col;

	printf("\n\n\n�ͦ��üƯx�}:\n");
	for (row = 0; row < dimension; row++)
	{
		for (col = 0; col < dimension; col++)
		{
			float r = rand() % 100;
			L[row * dimension + col] = r;
			printf("\t%f", r);
		}
		printf("\n");
	}

}

__global__ void nodiag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x != y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void diag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(float *A, float *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}
		}
	}

}

__global__ void set_zero(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

/*
* GPU�ϯx�}
*/
void inverse_matrix_GPU(const int n)
{
	printf("\n//////////////////////////////////////////////////////////////////\n");
	printf("\n/////////////////////////GPU�ϯx�}�}�l////////////////////////////\n");

	float *iL = new float[n*n];
	float *L = new float[n*n];

	matrix_gen(L, n);

	float *d_A, *d_L, *I, *dI;
	float time;
	cudaError_t err;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int ddsize = n*n*sizeof(float);

	dim3 threadsPerBlock(BLOCK_NUM, BLOCK_NUM);
	dim3 numBlocks((n + BLOCK_NUM - 1) / BLOCK_NUM, (n + BLOCK_NUM - 1) / BLOCK_NUM);

	err = cudaMalloc((void**)&d_A, ddsize);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMalloc((void**)&dI, ddsize);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	I = new float[n*n];

	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}

	err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

	//�ɶ��p�ɶ}�l
	cudaEventRecord(start, 0);

	// L^(-1)    
	for (int i = 0; i < n; i++){
		nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		diag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		gaussjordan << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		set_zero << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
	}



	err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }



	printf("�ϯx�}��:\n");
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			printf("\t%f", iL[i * n + j]);
		}
		printf("\n");
	}


	cudaFree(d_A);
	cudaFree(dI);

	float *c = new float[n*n];
	matmultCUDA_KSF_shared_pitch(L, n, iL, n, c, n, n);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("���Ҥϯx�}:\n");
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			printf("\t%f", c[i * n + j]);
		}
		printf("\n");
	}

	printf("GPU�ӶO�ɶ�: %f\n", time / CLOCKS_PER_SEC);
	delete[]I;
	delete[]L;
	delete[]iL;

}