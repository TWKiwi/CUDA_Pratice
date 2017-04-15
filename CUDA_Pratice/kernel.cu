#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <iostream>
#define DATA_SIZE 1048576
#define BLOCK_NUM 32
#define THREAD_NUM 256

int data[DATA_SIZE];
bool InitCUDA();
void GenerateNumbers(int *number, int size);
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
cudaDeviceProp prop;

int main()
{
	if (InitCUDA())
	{
		ArrayCompute();
		printf("(1).%dThreads��}����\n", THREAD_NUM);
		ArrayCompute_multiple_threads();
		printf("(2).(3).%dThreads �s��O����s������\n", THREAD_NUM);
		ArrayCompute_multiple_threads_continuous_access();
		printf("(4).%dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
		ArrayCompute_multiple_threads_blocks_continuous_access();
		printf("(5).Shared Memory %dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
		ArrayCompute_shared_multiple_threads_blocks_continuous_access();
		printf("(6).TreeSum alg. Shared Memory %dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
		ArrayCompute_shared_multiple_threads_blocks_continuous_access_treesum();
		printf("(7).��}TreeSum alg. Shared Memory %dThreads %dBlocks �s��O����s������\n", THREAD_NUM, BLOCK_NUM);
		ArrayCompute_shared_multiple_threads_blocks_continuous_access_better_treesum();
	}
	

	printf("\nDone!");
	getchar();
	return 0;
}

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
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	int* gpudata, *result;
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOfSquares << <1, 1, 0 >> >(gpudata, result);
	int sum;
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
 * Multiple Threads����
 */
void ArrayCompute_multiple_threads()
{
	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares_multiple_threads << <1, THREAD_NUM, 0 >> >(gpudata, result);

	int sum[THREAD_NUM];
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
* Multiple Threads����
* �s��O����s������
*/
void ArrayCompute_multiple_threads_continuous_access()
{
	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares_multiple_threads_continuous_access << <1, THREAD_NUM, 0 >> >(gpudata, result);

	int sum[THREAD_NUM];
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
* Multiple Threads Blocks����
* �s��O����s������
*/
void ArrayCompute_multiple_threads_blocks_continuous_access()
{
	
	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOfSquares_multiple_threads_blocks_continuous_access << <BLOCK_NUM, THREAD_NUM, 0 >> >(gpudata, result);
	int sum[THREAD_NUM * BLOCK_NUM];
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
* Shared Multiple Threads Blocks����
* �s��O����s������
*/
void ArrayCompute_shared_multiple_threads_blocks_continuous_access()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOfSquares_shared_multiple_threads_blocks_continuous_access << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result);
	int sum[THREAD_NUM * BLOCK_NUM];
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
* Shared Multiple Threads Blocks����
* �s��O����s������
* TreeSum alg.
*/
void ArrayCompute_shared_multiple_threads_blocks_continuous_access_treesum()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOfSquares_shared_multiple_threads_blocks_continuous_access_treesum << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result);
	int sum[THREAD_NUM * BLOCK_NUM];
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
* Shared Multiple Threads Blocks����
* �s��O����s������
* ��}TreeSum alg.
*/
void ArrayCompute_shared_multiple_threads_blocks_continuous_access_better_treesum()
{

	float timeValue;
	//-----------------------------------------------
	int* gpudata, *result;
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);
	GenerateNumbers(data, DATA_SIZE);
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_NUM * BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOfSquares_shared_multiple_threads_blocks_continuous_access_better_treesum << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result);
	int sum[THREAD_NUM * BLOCK_NUM];
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);
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
* ��}�ᥭ��[�`�{��
* multiple threads blocks
*/
__global__ static void sumOfSquares_multiple_threads_blocks_continuous_access(int *num, int* result)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int sum = 0;
	int i;
	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE;i += BLOCK_NUM * THREAD_NUM) {
		sum += num[i] * num[i];
	}
	result[bid * THREAD_NUM + tid] = sum;
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
	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE;i += BLOCK_NUM * THREAD_NUM) {
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
