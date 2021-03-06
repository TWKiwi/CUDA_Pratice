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
void bs(float* q, int lda, int n);
void matgenb(float* y, int lda, int n);
void matgen123(float* a, int lda, int n, float* z);
void printas(float* y, int lda, int n);
void aasd(float* a, float* y, int n, int lda);
void printasx(float* x, int lda, int n);

void inverse_matrix();
void inverse_matrix_CPU(int k);
void inverse_matrix_GPU(const int n);




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  雜七雜八              //////////////////////////////////////////
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
/////////////////////////////////////                  主進入點              //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{
	if (MODE == 0)
	{

		if (InitCUDA())
		{
			ArrayCompute();
			printf("(1).%dThreads改良版本\n", THREAD_NUM);
			ArrayCompute_multiple_threads();
			printf("(2)(3).%dThreads 連續記憶體存取版本\n", THREAD_NUM);
			ArrayCompute_multiple_threads_continuous_access();
			printf("(4).%dThreads %dBlocks 連續記憶體存取版本\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_multiple_threads_blocks_continuous_access();
			printf("(5).Shared Memory %dThreads %dBlocks 連續記憶體存取版本\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_shared_multiple_threads_blocks_continuous_access();
			printf("(6).TreeSum alg. Shared Memory %dThreads %dBlocks 連續記憶體存取版本\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_shared_multiple_threads_blocks_continuous_access_treesum();
			printf("(7).改良TreeSum alg. Shared Memory %dThreads %dBlocks 連續記憶體存取版本\n", THREAD_NUM, BLOCK_NUM);
			ArrayCompute_shared_multiple_threads_blocks_continuous_access_better_treesum();
			printf("//////////////FLOAT////////////////\n");
			printf("/////////浮點數的矩陣乘法//////////\n");
			int n = 0;
			printf("輸入矩陣大小(一個正整數):");
			scanf_s(" %d", &n);
			printf("(1).一般版本\n");
			FloatArrayMultiCompute(n);
			printf("(2).Kahan'sSummation Formula改良\n");
			FloatArrayMultiCompute_KSF(n);
			printf("(3).KSF改良 Shared Memory Pitch\n");
			FloatArrayMultiCompute_KSF_shared_pitch(n);
			printf("(4).反矩陣\n");
			inverse_matrix();


		}

		printf("\nDone!");
		system("pause");
		return 0;
	}
	else
	{
		/*
		 * 讓副執行續去啟動exe效果比直接用system()好，估計是因為直接用主執行續連續啟動應用程式的關係，負荷量過重了。
		 * 這是我的猜測。
		 */
		CMD = "cd ..\\Multi_Window_Display && start CUDA.exe && start CUDA_m_t.exe && start CUDA_m_t_c_a_(256).exe && start CUDA_m_t_c_a_(512).exe && start CUDA_m_t_b_c_a.exe && start CUDA_s_m_t_b_c_a.exe && start CUDA_s_m_t_b_c_a_t.exe && start CUDA_s_m_t_b_c_a_b_t.exe";
		std::thread thread(func_cmd);
		thread.join();

		return 0;
	}




}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
/////////////////////////////////////                  基本資訊              //////////////////////////////////////////
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
				printf("第%d張顯卡名稱 -> %s\n", count, prop.name);
				printf("--CUDA版本 -> %d.%d\n", prop.major, prop.minor);
				char msg[256];
				sprintf_s(msg, "--總Global Memory -> %.0f MBytes (%llu bytes)\n",
					(float)prop.totalGlobalMem / 1048576.0f, (unsigned long long) prop.totalGlobalMem);
				printf("%s", msg);
				printf("--%2d 個 Multiprocessors\n", prop.multiProcessorCount);
				printf("--每個Multiprocessors裡最大執行續數量:%d\n", prop.maxThreadsPerMultiProcessor);
				printf("--每個Block裡最大執行緒數量:%d\n", prop.maxThreadsPerBlock);
				printf("--GPU 最大時脈: %.0f MHz (%0.2f GHz)\n", prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
				printf("--最大記憶體時脈: %.0f Mhz (%0.2f GHz)\n", prop.memoryClockRate * 1e-3f, prop.memoryClockRate * 1e-6f);
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
/////////////////////////////////////                  正整數                //////////////////////////////////////////
/////////////////////////////////////                  平方和                //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * 生亂數串列
 */
void GenerateNumbers(int *number, int size)
{
	for (int i = 0; i < size; i++)
	{
		number[i] = rand() % 10;
	}
}

/*
 * 1 Thread版本
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
	printf("--執行時間 (GPU): %f\n", float(timeValue) / CLOCKS_PER_SEC);
	//-----------------------------------------------

	sum = 0;
	clock_t cpu_time = clock();
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += data[i] * data[i];
	}
	printf("--sum (CPU): %d\n", sum);
	printf("--執行時間 (CPU): %f\n", float(clock() - cpu_time) / CLOCKS_PER_SEC);

}
/*
* 原版平方加總程式
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
 * Multiple Threads版本
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
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // 只適用於32位元資料前提 (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--執行時間 (GPU): %f | 時脈: %fMHz | 記憶體頻寬:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* 改良後平方加總程式
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
* Multiple Threads版本
* 連續記憶體存取版本
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
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // 只適用於32位元資料前提 (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--執行時間 (GPU): %f | 時脈: %fMHz | 記憶體頻寬:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* 改良後平方加總程式
* multiple threads 連續記憶體存取
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
* Multiple Threads Blocks版本
* 連續記憶體存取版本
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
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // 只適用於32位元資料前提 (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--執行時間 (GPU): %f | 時脈: %fMHz | 記憶體頻寬:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* 改良後平方加總程式
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
* Shared Multiple Threads Blocks版本
* 連續記憶體存取版本
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
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // 只適用於32位元資料前提 (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--執行時間 (GPU): %f | 時脈: %fMHz | 記憶體頻寬:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* 改良後平方加總程式
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
* Shared Multiple Threads Blocks版本
* 連續記憶體存取版本
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
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // 只適用於32位元資料前提 (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--執行時間 (GPU): %f | 時脈: %fMHz | 記憶體頻寬:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);
}
/*
* 改良後平方加總程式
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
* Shared Multiple Threads Blocks版本
* 連續記憶體存取版本
* 改良TreeSum alg.
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
	float memory_bandwidth = 4 / (float(timeValue) / CLOCKS_PER_SEC); // 只適用於32位元資料前提 (1024 * 1024 * 32(bit)) / 8(bit -> byte) * 1024(byte -> kb) * 1024(kb -> mb)
	printf("--sum (GPU): %d\n", final_sum);
	printf("--執行時間 (GPU): %f | 時脈: %fMHz | 記憶體頻寬:%f MB/s\n", float(timeValue) / CLOCKS_PER_SEC, clock_cycle, memory_bandwidth);

}
/*
* 改良後平方加總程式
* shared multiple threads blocks
* 改良TreeSum alg.
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
/////////////////////////////////////                  浮點數                //////////////////////////////////////////
/////////////////////////////////////                 矩陣相乘               //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
* 生成亂數浮點數矩陣
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
* CPU版矩陣乘法
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
* CPU驗證矩陣相乘結果
* 印出最大誤差值，平均誤差值
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
	printf("最大誤差值: %g 平均誤差值: %g\n", max_err, average_err / (n * n));

}

/*
*浮點數矩陣乘法
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
	clock_t cpu_time_start = clock();//CPU計時開始
	matmult(a, n, b, n, d, n, n);
	cpu_time_use_per_sec = (float)(clock() - cpu_time_start) / CLOCKS_PER_SEC;//CPU計時結束並結算
	compare_mat(c, n, d, n, n);
	double sec = (double)time / CLOCKS_PER_SEC;
	printf("CPU Time used: %f\n", cpu_time_use_per_sec);
	printf("GPU Time used: %f (%f GFLOPS)\n", sec, 2.0 * n * n * n / (sec * 1E9));
}

/*
 * 有點不太明白為什麼這邊老師要特地把cudaMalloc、cudaMemcpy2D、cudaFree考慮進去計算時間，之前就不考慮...
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
 * 矩陣相乘的Kernel函式
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
 * 浮點數矩陣乘法
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
* 有點不太明白為什麼這邊老師要特地把cudaMalloc、cudaMemcpy2D、cudaFree考慮進去計算時間，之前就不考慮...
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
* 矩陣相乘的Kernel函式
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
			//這邊真的看不懂在幹嘛...
			float r;
			y -= a[row * lda + i] * b[i * ldb + column];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		c[row * ldc + column] = t; //這行是我加上去的，因為不加這行怎麼算誤差值都會是1，主要問題點還是上面...
	}
	//這段寫法比較好理解，可是跟老師得稍微不同...
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
* 浮點數矩陣乘法
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
* 有點不太明白為什麼這邊老師要特地把cudaMalloc、cudaMemcpy2D、cudaFree考慮進去計算時間，之前就不考慮...
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
* 矩陣相乘的Kernel函式
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
/////////////////////////////////////                  反矩陣                //////////////////////////////////////////
/////////////////////////////////////           GPU手動輸入值還未完成        //////////////////////////////////////////
/////////////////////////////////////                                        //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void inverse_matrix()
{
	int i, j, k;
	char ans;
	printf("\n輸入矩陣大小 : ");
	scanf(" %d", &k);
	

	inverse_matrix_GPU(k);
	inverse_matrix_CPU(k);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////開始實作做CPU反矩陣///////////////////////////////////////////////////

/*
* 反矩陣
*/
void inverse_matrix_CPU(int n)
{
	printf("\n//////////////////////////////////////////////////////////////////\n");
	printf("\n/////////////////////////CPU反矩陣開始////////////////////////////\n");
	float *a, *b, *c, *d, *x, *y, *z, *q;


	a = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	x = (float*)malloc(sizeof(float)* n * n);
	y = (float*)malloc(sizeof(float)* n * n);
	z = (float*)malloc(sizeof(float)* n * n);
	q = (float*)malloc(sizeof(float)* n * n);
	bs(q, n, n);
	//matgen(a, n, n);
	
	clock_t start = clock();
	printf("\n產生亂數矩陣: \n");
	matgen123(a, n, n, z);
	matgenb(y, n, n);//反矩陣
	aasd(a, y, n, n);
	matmult(a, n, y, n, c, n, n);
	printf("\n產生反矩陣: \n");
	printas(c, n, n);
	matmult(c, n, z, n, x, n, n);

	clock_t end = clock();

	printf("\n驗證反矩陣: \n");
	printasx(x, n, n);

	printf("耗時%f秒\n\n", (float)(end - start) / CLOCKS_PER_SEC);
}

void bs(float* q, int lda, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i == j){
				q[i * lda + j] = 1;
			}
			else
			{
				q[i * lda + j] = 0;
			}
		}
	}
}
//反矩陣
void matgenb(float* y, int lda, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i == j)
				y[i * lda + j] = 1;
			else
				y[i * lda + j] = 0;
			//printf("\t%f  ",y[i * lda + j]);
		}
	}
}

void matgen123(float* a, int lda, int n, float* z)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			a[i * lda + j] = (float)(rand() % 100);
			z[i * lda + j] = a[i * lda + j];
			printf("\t%.5f  ", z[i * lda + j]);
		}
		printf("\n");
	}
}

void printas(float* y, int lda, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{

			printf("\t%.5f  ", y[i * lda + j]);
		}
		printf("\n");
	}
}

void aasd(float* a, float* y, int n, int lda)
{
	double temp;
	for (int k = 0; k<n; k++)
	{
		temp = 0;
		for (int j = 0; j<n; j++)
		{
			temp += a[k *lda + j];
			if (a[k *lda + k] != 1)
			{
				temp = a[k *lda + k];
				for (int i = 0; i<n; i++)
				{
					a[k *lda + i] = a[k *lda + i] / temp;//亂數陣列
					y[k *lda + i] = y[k *lda + i] / temp;
				}
			}

			for (int i = 0; i<n; i++)
			{
				if (a[i *lda + k] != 0 && i != k)
				{
					temp = a[i *lda + k];
					for (int j = 0; j<n; j++)
					{
						a[i *lda + j] = a[i *lda + j] - (a[k *lda + j] * temp);
						y[i *lda + j] = y[i *lda + j] - (y[k *lda + j] * temp);
					}
				}
			}
		}
	}
}

void printasx(float* x, int lda, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			printf("\t%.5f  ", x[i * lda + j]);
		}
		printf("\n");
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////開始實作做GPU反矩陣///////////////////////////////////////////////////
/*
 * 修改自
 * https://github.com/ZhengzhongSun/Matrix-Inversion-with-CUDA
 */
void matrix_gen(float *L, int dimension){
	int row, col;

	printf("\n\n\n生成亂數矩陣:\n");
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
* GPU反矩陣
*/
void inverse_matrix_GPU(const int n)
{
	printf("\n//////////////////////////////////////////////////////////////////\n");
	printf("\n/////////////////////////GPU反矩陣開始////////////////////////////\n");

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

	//時間計時開始
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



	printf("反矩陣為:\n");
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

	printf("驗證反矩陣:\n");
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			printf("\t%f", c[i * n + j]);
		}
		printf("\n");
	}

	printf("GPU耗費時間: %f\n", time / CLOCKS_PER_SEC);
	delete[]I;
	delete[]L;
	delete[]iL;

}