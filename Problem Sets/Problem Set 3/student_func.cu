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

/*
__device__ float d_max;

__global__ void find_max2(const float* const d_logLuminance, 
            const size_t numRows, const size_t numCols)
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
        atomicMaxf(&d_max, sh_array[idx]);
        //max_logLum[blockIdx.x] = sh_array[idx];
    }
   
}
*/

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


__global__ void histo_phase_1 (const float* const d_logLuminance, int *hist_array,
        const size_t numRows, const size_t numCols, const size_t numBins, 
        const float logLumMin, const float logLumRange)
{
    __shared__ extern int sh_hist[];
    
    for(int i = threadIdx.x; i<numBins; i+=blockDim.x){
        sh_hist[i] = 0;
    }
    
    if(blockIdx.x*blockDim.x + threadIdx.x > numRows*numCols)
        return;

    unsigned int bin = min(
            static_cast<unsigned int>(numBins - 1),
            static_cast<unsigned int>((d_logLuminance[threadIdx.x] - logLumMin) / logLumRange * numBins));

    atomicAdd(&sh_hist[bin], 1);

    __syncthreads();

    for(int i = threadIdx.x; i<numBins; i+=blockDim.x){
        hist_array[blockIdx.x*numBins + i] = sh_hist[i];
    }

}

__global__ void histo_phase_2 (int * hist_array,const int numHists, const size_t numBins)
{
    
    //int limit = numHists;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    for(int s = 1; s < numHists; s*=2){
        int other = idx + s; 
        if(idx % (2*s) == 0 && other < numHists){
            for(int i =0; i< numBins; i++){
                hist_array[idx * numBins+i] += hist_array[other*numBins +i];
            }
        }
        __syncthreads();
    }


}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    // Assuming that I don't have to allocate any other memeory in
    // the device
    //const dim3 blockSize(32, 16, 1);
    //const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, 
    //                    (numRows + blockSize.y - 1) / blockSize.y, 1);
    
    // Step 1:
    // Use reduce(d_logLuminance, min) in order to find the minimum value
    // and reduce(d_logLuminance, max) in order to find the maximum value
    
    int threads = 1024;
    int blocks = (numCols * numRows +2*threads -1) / (2*threads);
    //std::cout << "Num of blocks is " << blocks << std::endl;
    ssize_t min_max_array_size = blocks * sizeof(float);
    float * d_minArray;
    float * d_maxArray;
    float * h_minArray = (float *) malloc (min_max_array_size);
    float * h_maxArray = (float *) malloc (min_max_array_size);
    
    float * h_logLuminance = (float *) malloc (numRows * numCols * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance,  numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    
    
    float ref_min = h_logLuminance[0];
    float ref_max = h_logLuminance[0];
    for(int i =1; i< numCols*numRows; i++){
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
    

    for(int i =0; i< 10; i++)
        printf("Real Hist[%d] = %d\n", i, histo[i]);

    
    //printf("real min is %f\n", min);
    //printf("real max is %f\n", max);
    
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
    for(int i =1; i< blocks; i++){
        min_logLum = std::min(min_logLum, h_minArray[i]);
        max_logLum = std::max(max_logLum, h_maxArray[i]);
    }

    // End of step 1
    
    // Find range
    float range = max_logLum - min_logLum;
    // End of step 2
    
    threads = 512;
    blocks = (numCols * numRows + threads -1) / (threads);
    int numHists = blocks;
    int *d_hist_array;
    checkCudaErrors(cudaMalloc(&d_hist_array,   blocks * numBins * sizeof(int)));
    
    histo_phase_1<<<blocks, threads, numBins * sizeof(int)>>> (d_logLuminance, d_hist_array, numRows, 
                                        numCols, numBins, min_logLum, range);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // now we need to combine the partial histograms
    
    threads = numBins;
    blocks = 1;//(numBins + threads - 1) / threads;

    histo_phase_2<<<blocks, threads>>>(d_hist_array,numHists, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
    int * h_hist = new int[numBins];
    checkCudaErrors(cudaMemcpy(h_hist, d_hist_array,  numBins*sizeof(int), cudaMemcpyDeviceToHost));
    
    for(int i =0; i< 10; i++)
        printf("Hist[%d] = %d\n", i, h_hist[i]);


    //std::cout << "numRows " << numRows << "\n";
    //std::cout << "numCols " << numCols << "\n";
    //std::cout << "numBins " << numBins << "\n";
    // generate one histogram per block, then use atomicAdd
    // to add them to the global histogram

    //printf("min = %f\n", min_logLum);
    //printf("max = %f\n", max_logLum);
    //printf("max2 = %f\n", d_max);
    //std::cout << "min = " << min_logLum << std::endl;
    //std::cout << "max = " << max_logLum << std::endl;

    //TODO
    /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    /*
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int r = threadIdx.y + blockIdx.y * blockDim.y;

    
    logLumMin = h_logLuminance[0];
    logLumMax = h_logLuminance[0];

    //Step 1
    //first we find the minimum and maximum across the entire image
    for (size_t i = 1; i < numCols * numRows; ++i) {
        logLumMin = std::min(h_logLuminance[i], logLumMin);
        logLumMax = std::max(h_logLuminance[i], logLumMax);
    }

    //Step 2
    float logLumRange = logLumMax - logLumMin;

    //Step 3
    //next we use the now known range to compute
    //a histogram of numBins bins
    unsigned int *histo = new unsigned int[numBins];

    for (size_t i = 0; i < numBins; ++i) histo[i] = 0;

    for (size_t i = 0; i < numCols * numRows; ++i) {
        unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((h_logLuminance[i] - logLumMin) / logLumRange * numBins));
        histo[bin]++;
    }

    //Step 4
    //finally we perform and exclusive scan (prefix sum)
    //on the histogram to get the cumulative distribution
    h_cdf[0] = 0;
    for (size_t i = 1; i < numBins; ++i) {
        h_cdf[i] = h_cdf[i - 1] + histo[i - 1];
    }
    */


}
