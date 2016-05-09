/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"
//__device__ void swap(float &a, float &b){float c = a; a=b; b=c;}

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ float atomicMinf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                __float_as_int(val));
    }
    return __int_as_float(old);
}

__global__ void find_min(const float* const d_logLuminance, 
        float *min_logLum, const size_t numRows, const size_t numCols)
{
    
    int index = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    int imgSize = numCols*numRows;
    __shared__ extern float sh_array[];
    
    if(index < imgSize)
        sh_array[2*threadIdx.x] = d_logLuminance[index];
    
    if(index+1 < imgSize)
        sh_array[2*threadIdx.x+1] = d_logLuminance[index+1];
    
    __syncthreads();
    
    int limit = min(2*blockDim.x, imgSize - 2*blockIdx.x*blockDim.x);
    int idx = 2*threadIdx.x;
    
    for(int s = 1; s< limit; s*=2){
        int other = idx + s; 
        if(idx % (2*s) == 0 && other < limit){
            //float temp = sh_array[idx];
            sh_array[idx] = min(sh_array[idx], sh_array[other]);
        }
        __syncthreads();
    }

    __syncthreads();
    
    if(threadIdx.x==0){
        //printf("min is %f\n", sh_array[0]);
        min_logLum[blockIdx.x] = sh_array[idx];
    }
   
}


__global__ void find_max(const float* const d_logLuminance, 
        float *max_logLum, const size_t numRows, const size_t numCols)
{
    
    int index = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    int imgSize = numCols*numRows;
    __shared__ extern float sh_array[];
    
    if(index < imgSize)
        sh_array[2*threadIdx.x] = d_logLuminance[index];
    
    if(index+1 < imgSize)
        sh_array[2*threadIdx.x+1] = d_logLuminance[index+1];
    
    __syncthreads();
    
    int limit = min(2*blockDim.x, imgSize - 2*blockIdx.x*blockDim.x);
    int idx = 2*threadIdx.x;
    
    for(int s = 1; s< limit; s*=2){
        int other = idx + s; 
        if(idx % (2*s) == 0 && other < limit){
            sh_array[idx] = max(sh_array[idx], sh_array[other]);
        }
        __syncthreads();
    }

    __syncthreads();
    
    if(threadIdx.x==0){
        max_logLum[blockIdx.x] = sh_array[idx];
    }
   
}



__global__ void find_min_max(const float* const d_logLuminance, 
        float *min_logLum, float *max_logLum, const size_t numRows,
        const size_t numCols)
{
    
    int index = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    int imgSize = numCols*numRows;
    __shared__ extern float sh_array[];
    
    if(index < imgSize)
        sh_array[2*threadIdx.x] = d_logLuminance[index];
    
    if(index+1 < imgSize)
        sh_array[2*threadIdx.x+1] = d_logLuminance[index+1];
    
    __syncthreads();
    
    int limit = min(2*blockDim.x, imgSize - 2*blockIdx.x*blockDim.x);
    int idx = 2*threadIdx.x;
    
    for(int s = 1; s< limit; s*=2){
        int other = idx + s; 
        if(idx % (2*s) == 0 && other < limit){
            float temp = sh_array[idx];
            sh_array[idx] = min(sh_array[idx], sh_array[other]);
            if(s==1)
                sh_array[idx+1] = max(temp, sh_array[other]);
            else if(other +1 < limit)
                sh_array[idx+1] = max(sh_array[idx+1],sh_array[other+1] );
            else 
                sh_array[idx+1] = max(sh_array[idx+1],sh_array[other] );
        }
        __syncthreads();
    }

    __syncthreads();
    
       if(threadIdx.x==0){
        //printf("min is %f\n", sh_array[0]);
        min_logLum[blockIdx.x] = sh_array[idx];
        max_logLum[blockIdx.x] = sh_array[idx+1];
    }
}


__global__ void histo_phase_1 (const float* const d_logLuminance, unsigned int * const hist_array,
        const size_t numRows, const size_t numCols, const size_t numBins, 
        const float logLumMin, const float logLumRange)
{
    __shared__ extern unsigned int sh_hist[];
    
    for(size_t i = threadIdx.x; i<numBins; i+=blockDim.x){
        sh_hist[i] = 0;
    }
    
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid > numRows*numCols)
        return;

    unsigned int bin = min(
            static_cast<unsigned int>(numBins - 1),
            static_cast<unsigned int>((d_logLuminance[gid] - logLumMin) / logLumRange * numBins));

    atomicAdd(&sh_hist[bin], 1);

    __syncthreads();

    for(size_t i = threadIdx.x; i<numBins; i+=blockDim.x){
        hist_array[blockIdx.x*numBins + i] = sh_hist[i];
    }

}

__global__ void histo_phase_2 (unsigned int * const hist_array,const size_t numHists, const size_t numBins)
{
    
    //int limit = numHists;
    size_t idx = blockDim.x*blockIdx.x + threadIdx.x;
    for(size_t s = 1; s < numHists; s*=2){
        size_t other = idx + s; 
        if(idx % (2*s) == 0 && other < numHists){
            for(size_t i =0; i< numBins; i++){
                hist_array[idx * numBins+i] += hist_array[other*numBins +i];
            }
        }
        __syncthreads();
    }
}

__global__ void histo_phase_2_fast (unsigned int * const hist_array,const size_t numHists, const size_t numBins, const size_t s)
{
    
    size_t idx = 2*blockDim.x * blockIdx.x + threadIdx.x;
    size_t line = 2*blockIdx.x;
    size_t other = idx + s*numBins; 
    
    if(line % (2*s) == 0 && other < numHists*numBins){
        hist_array[idx] += hist_array[other];
    }
}



__global__ void histo_scan (unsigned int * const d_cdf, const unsigned int * const d_hist,
                            size_t numBins)
{
    extern __shared__ unsigned int sh_cdf[];

    int tid = 2*threadIdx.x;
    int offset = 1;

    if(tid < numBins)
        sh_cdf[tid] = d_hist[tid];
    
    if(tid +1 < numBins)
        sh_cdf[tid+1] = d_hist[tid+1];

    for (int d = numBins>>1; d > 0; d>>= 1)
    {
        __syncthreads();

        if (threadIdx.x < d)
        {
            int ai = offset * (tid +1) - 1;
            int bi = offset * (tid +2) - 1;
            sh_cdf[bi] += sh_cdf[ai];
        }
        offset *= 2;
    }

    if (tid == 0) 
        sh_cdf[numBins -1] = 0;

    for (int d = 1; d < numBins; d*=2)
    {
        offset >>= 1;
        __syncthreads();

        if(threadIdx.x < d)
        {
            int ai = offset * (tid +1) - 1;
            int bi = offset * (tid +2) - 1;
            unsigned int t = sh_cdf[ai];
            sh_cdf[ai] = sh_cdf[bi];
            sh_cdf[bi] += t;
        }
    }

    __syncthreads();

    if(tid < numBins)
        d_cdf[tid] = sh_cdf[tid];
    
    if(tid +1 < numBins)
        d_cdf[tid+1] = sh_cdf[tid+1];

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

    
    
    /*
    Serial part: just for testing reasons
    float * h_logLuminance = (float *) malloc (numRows * numCols * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance,  numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    
    
    float ref_min = h_logLuminance[0];
    float ref_max = h_logLuminance[0];
    for(size_t i =1; i< numCols*numRows; i++){
        ref_min = std::min(ref_min, h_logLuminance[i]);
        ref_max = std::max(ref_max, h_logLuminance[i]);
    }
    float ref_range = ref_max - ref_min;
    
    unsigned int *histo = new unsigned int[numBins];

    for (size_t i = 0; i < numBins; ++i) histo[i] = 0;

    for (size_t i = 0; i < numCols * numRows; ++i) {
        unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((h_logLuminance[i] - ref_min) / ref_range * numBins));
        histo[bin]++;
    }

    unsigned int *h_cdf_ref = new unsigned int[numBins];

    h_cdf_ref[0] = 0;
    for (size_t i = 1; i < numBins; ++i) {
        h_cdf_ref[i] = h_cdf_ref[i - 1] + histo[i - 1];
    }

    for(size_t i =0; i< numBins; i+=64)
        printf("Real CDF[%d] = %d\n", i, h_cdf_ref[i]);

    */

    size_t threads = 128;
    size_t blocks = (numCols * numRows +2*threads -1) / (2*threads);
    size_t min_max_array_size = blocks * sizeof(float);
    float * d_minArray;
    float * d_maxArray;
    float * h_minArray = (float *) malloc (min_max_array_size);
    float * h_maxArray = (float *) malloc (min_max_array_size);

    checkCudaErrors(cudaMalloc(&d_minArray,    min_max_array_size));
    checkCudaErrors(cudaMalloc(&d_maxArray,    min_max_array_size));
    
    find_min_max<<<blocks, threads, 2 * threads * sizeof(float) >>>(d_logLuminance, d_minArray, 
            d_maxArray, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /*
    find_min<<<blocks, threads, 2*threads * sizeof(float)>>> (d_logLuminance, d_minArray, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    find_max<<<blocks, threads, 2*threads * sizeof(float)>>> (d_logLuminance, d_maxArray, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    */
    
    checkCudaErrors(cudaMemcpy(h_minArray, d_minArray,  min_max_array_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_maxArray, d_maxArray,  min_max_array_size, cudaMemcpyDeviceToHost));
    
    min_logLum = h_minArray[0];
    max_logLum = h_maxArray[0];

    // num of blocks is quite small
    for(size_t i =1; i< blocks; i++){
        min_logLum = std::min(min_logLum, h_minArray[i]);
        max_logLum = std::max(max_logLum, h_maxArray[i]);
    }

    // End of step 1
    
    // Start of Step 2: Find range
    float range = max_logLum - min_logLum;
    // End of Step 2
    
    // Start of Step 3: Calculate histogram
    
    threads = 1024;
    blocks = (numCols * numRows + threads -1) / (threads);
    size_t numHists = blocks;
    unsigned int *d_hist_array;
    checkCudaErrors(cudaMalloc(&d_hist_array,   blocks * numBins * sizeof(unsigned int)));
    
    histo_phase_1<<<blocks, threads, numBins * sizeof(unsigned int)>>> (d_logLuminance, d_hist_array, numRows, 
                                        numCols, numBins, min_logLum, range);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // now we need to combine the partial histograms
        
    threads = numBins;
    //blocks = 1;//(numBins + threads - 1) / threads;
    //threads = numBins; 
    blocks = (numHists+1) / 2;//(numBins * numHists + 2* threads -1 ) / (2* threads);
    //histo_phase_2<<<blocks, threads>>>(d_hist_array,numHists, numBins);
    for(size_t s = 1; s < numHists; s*=2){
        histo_phase_2_fast<<<blocks, threads>>>(d_hist_array,numHists, numBins, s);
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // End of Step 3

    // Start of Step 4: calculate CDF

    threads = (numBins + 1) / 2;
    blocks = 1;//(numBins + 2*threads -1) / (2*threads);

    // maybe we need to initiate half of the blocks
    histo_scan<<<blocks, threads, 2 * threads * sizeof(unsigned int)>>>(d_cdf, d_hist_array, numBins); 
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // End of Step 4
    

}
