#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdlib>

#define NUM_ELEMENTS 4096
#define SHOW_ELAPSED_TIME 1

//Coefficients for FIR
#define FIR_SIZE 64
const int fir_size_in_bytes = FIR_SIZE * sizeof(float);
__constant__ float fir_coef [FIR_SIZE];
float cpu_fir_coef[FIR_SIZE] = {0.0};

// Declare arrays dynamically
float *cpu_I_in_buffer;
float *cpu_I_result_buffer;
float *cpu_Q_in_buffer;
float *cpu_Q_result_buffer;

//Default values, can be overwritten by providing command line arguments
unsigned int array_size;
unsigned int array_size_in_bytes;
unsigned int num_threads = FIR_SIZE;
unsigned int num_blocks = NUM_ELEMENTS;

//Function to copy data into shared memory. Includes thread sync
__device__ 
void copy_data_to_shared(float * src, float * dst, const unsigned int tid)
{
    // Copy data
    dst[tid] = src[tid];

    // Sync threads before accessing shared memory
    __syncthreads();
}

//Function to copy data out of shared memory. Includes thread sync
__device__ 
void copy_data_from_shared(float * src, float * dst, const unsigned int tid)
{
    // Sync threads before accessing shared memory
    __syncthreads();

    // Copy data
    dst[tid] = src[tid];
}


// Custom complex multiplication kernel
__device__
void cMult(const float Ai, const float Aq, const float Bi, const float Bq, float* Ri, float* Rq)
{
    *Ri = Ai*Bi - Aq*Bq;
    *Rq = Ai*Bq + Aq*Bi;
}

//Method to quickly sum an array and put the result at the begining of the array
//http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__device__
void sum_array(float * sdata, const unsigned int blockSize, const unsigned int tid)
{
//Temporary fix
#if 1
    if(tid==0)
    {
        for(unsigned int i=1; i<blockSize; i++)
            sdata[0] += sdata[i];
    }
    return;
#endif
//TODO: Figure out why this parallel adder doesn't work
    if(blockSize >= 512) 
    {
        if(tid < 256) 
            sdata[tid] += sdata[tid + 256];
        __syncthreads(); 
    }
    
    if(blockSize >= 256) 
    {
        if(tid < 128) 
            sdata[tid] += sdata[tid + 128]; 
        __syncthreads(); 
    }

    if(blockSize >= 128) {
        if(tid <  64)  
            sdata[tid] += sdata[tid +   64]; 
        __syncthreads(); 
    }
    
    if(tid < 32)
    {
        if (blockSize >=  64)
            sdata[tid] += sdata[tid + 32];
        if (blockSize >=  32)
            sdata[tid] += sdata[tid + 16];
        if (blockSize >=  16)
            sdata[tid] += sdata[tid +  8];
        if (blockSize >=   8)
            sdata[tid] += sdata[tid +  4]; 
        if (blockSize >=   4)
            sdata[tid] += sdata[tid +  2]; 
        if (blockSize >=   2)
            sdata[tid] += sdata[tid +  1]; 
    }
    __syncthreads();
}

// Perform a frequency shift via complex multiply
// Parameters:
// I_in, Q_in, I_out, Q_out: 4 data buffers, all must be the same length
// n0: Used to calculate the phase of the first point of the mixing signal
// freq_shift: Frequency to shift in Hz
// Fs: sample frequency in Hz
__global__
void freq_shift(float * I_in, float * Q_in, float * I_out, float * Q_out, const unsigned int n0, const float freq_shift, const float Fs)
{
    //Who am I?
    //const unsigned int thread_id = threadIdx.x;
    //const unsigned int block_id = blockIdx.x;
    const unsigned int global_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    float I_shift;
    float Q_shift;
    float theta_nopi = 2.0*freq_shift*(n0 + global_index)/Fs;   

    sincospif(theta_nopi, &Q_shift, &I_shift);
    cMult(I_in[global_index], Q_in[global_index], I_shift, Q_shift, &I_out[global_index], &Q_out[global_index]);    
}

// FIR based decimation
__global__
void decimate(float * input_buffer, float * output_buffer, const unsigned int decimation_factor)
{
    __shared__ float conv[FIR_SIZE];

    //Who am I?
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_id = blockIdx.x;
	//const unsigned int global_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    //Perform the convolution as a copy from global (num samples) to shared (FIR width)
    float sample = 0.0;
    int sample_index = block_id - thread_id;
    if(sample_index >= 0)
        sample = input_buffer[sample_index];    
    conv[thread_id] = sample*fir_coef[thread_id];
    __syncthreads();

    //Sum results vector using loop unrolling and shared memory
    sum_array(conv, blockDim.x, thread_id);
    
    //Decimate
    if(thread_id == 0)
    {
        if((block_id % decimation_factor) == 0)
            output_buffer[block_id / decimation_factor] = conv[0];
    }
}

// main_sub0 : Method to copy an input buffer into cuda and copy the results out
void main_sub0()
{
	// Declare pointers for GPU based params
    //float *gpu_input_buffer;
    //float *gpu_output_buffer;
    float *gpu_I_in_buffer;
    float *gpu_I_mixed_buffer;
    float *gpu_I_result_buffer;
    float *gpu_Q_in_buffer;
    float *gpu_Q_mixed_buffer;
    float *gpu_Q_result_buffer;

    // Allocate memory in the GPU
    cudaMalloc((void **)&gpu_I_in_buffer, array_size_in_bytes);
    cudaMalloc((void **)&gpu_I_mixed_buffer, array_size_in_bytes);
    cudaMalloc((void **)&gpu_I_result_buffer, array_size_in_bytes);
    cudaMalloc((void **)&gpu_Q_in_buffer, array_size_in_bytes);
    cudaMalloc((void **)&gpu_Q_mixed_buffer, array_size_in_bytes);
    cudaMalloc((void **)&gpu_Q_result_buffer, array_size_in_bytes);

    //Copy Constant data
    cudaMemcpyToSymbol(fir_coef, &cpu_fir_coef, fir_size_in_bytes);

    //Create stream
    //cudaStream_t stream1, stream2;
    //cudaStreamCreate(&stream1);
    //cudaStreamCreate(&stream2);

#if SHOW_ELAPSED_TIME
    float ms;

    // Setup Start and Stop event
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Start timer
    cudaEventRecord(startEvent, 0);
#endif

    // Copy data from CPU to GPU
	cudaMemcpy(gpu_I_in_buffer, cpu_I_in_buffer, array_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_Q_in_buffer, cpu_Q_in_buffer, array_size_in_bytes, cudaMemcpyHostToDevice);

    freq_shift<<<num_blocks/32, 32>>>(gpu_I_in_buffer, gpu_Q_in_buffer, gpu_I_mixed_buffer, gpu_Q_mixed_buffer, 0, -10000, 1e6);

    decimate<<<num_blocks, num_threads>>>(gpu_I_mixed_buffer, gpu_I_result_buffer, 1);
    decimate<<<num_blocks, num_threads>>>(gpu_Q_mixed_buffer, gpu_Q_result_buffer, 1);
    
    // Copy results from GPU to CPU	
	cudaMemcpy(cpu_I_result_buffer, gpu_I_result_buffer, array_size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_Q_result_buffer, gpu_Q_result_buffer, array_size_in_bytes, cudaMemcpyDeviceToHost);

#if SHOW_ELAPSED_TIME
    // Stop timer
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Elapsed Time: %f ms\n", ms);

    // Destroy timer
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
#endif

    //Destory streams
    //cudaStreamDestroy(stream1);
    //cudaStreamDestroy(stream2);

    // Free the arrays on the GPU
    cudaFree(gpu_I_in_buffer);
    cudaFree(gpu_I_mixed_buffer);
    cudaFree(gpu_I_result_buffer);
    cudaFree(gpu_Q_in_buffer);
    cudaFree(gpu_Q_mixed_buffer);
    cudaFree(gpu_Q_result_buffer);

}

//main : parse command line arguments and run GPU code
int main(int argc, char *argv[])
{
    // Argument parsing using getopt
    // http://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html#Example-of-Getopt
    // -b <int> sets the number of GPU blocks
    // -t <int> sets the nubmer of GPU threads
    // -v Sets verbose flag - shows math results

    int c;
    //bool showMathResults = false;
    
    while ((c = getopt (argc, argv, "b:t:v")) != -1)
    switch (c)
    {
        case 'b':
            num_blocks = atoi(optarg);
            break;
        case 't':
            num_threads = atoi(optarg);
            break;
        case 'v':
            //showMathResults = true;
            break;
        default:
            printf("USAGE:\n-b <int> GPU blocks\n-t <int> GPU threads (each block)\n-v Verbose\n");
            return EXIT_SUCCESS;
    }

    printf("Blocks: %d\nThreads: %d\n", num_blocks, num_threads);

    // Calculate buffer size
    array_size = num_blocks; // * num_threads;
    array_size_in_bytes = sizeof(float) * (array_size);

    // Allocate memory on the CPU
    cpu_I_in_buffer = new float[array_size];
    cpu_I_result_buffer = new float[array_size];
    cpu_Q_in_buffer = new float[array_size];
    cpu_Q_result_buffer = new float[array_size];

    //Load fir
    FILE * iFile;
    iFile = fopen("fir_dec_2_taps_64.txt", "r");
    for(unsigned int i=0; i<FIR_SIZE; i++)
        fscanf(iFile, "%f\r\n", &cpu_fir_coef[i]);
    fclose(iFile);

    // Generate data to be processed
#if 0
    for(unsigned int i=0; i<array_size; i++)
    {
        cpu_I_in_buffer[i] = 0.0;
        cpu_Q_in_buffer[i] = 0.0;
    }
    //cpu_I_in_buffer[NUM_ELEMENTS/2] = 1.0;
    cpu_I_in_buffer[0] = 1.0;
#else
    float I, Q;
    iFile = fopen("inputIQ.txt", "r+");   
    for(unsigned int i=0; i<array_size; i++)
    {
        fscanf(iFile, "%f,%f\r\n", &I, &Q);
        cpu_I_in_buffer[i] = I;
        cpu_Q_in_buffer[i] = Q;
    }
    fclose(iFile);
#endif

    // Run
    main_sub0();

#if 0
    // Output results
    if(showMathResults)
    {
	    for(unsigned int i = 0; i < array_size; i++)
	    {
		    printf("Calc: %.5f => %.5f\n", cpu_input_buffer[i], cpu_output_buffer[i]);
	    }
    }
    printf("\n");
#else
    FILE * oFile;
    oFile = fopen("outputIQ.txt", "w+");
    for(unsigned int i=0; i<array_size; i++)
    {
        fprintf(oFile, "%f,%f\r\n", cpu_I_result_buffer[i], cpu_Q_result_buffer[i]);
    }
    fclose(oFile);
#endif

    // Done
	return EXIT_SUCCESS;
}
