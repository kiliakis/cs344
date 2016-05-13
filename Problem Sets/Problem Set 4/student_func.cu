//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include "stdio.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


   /*
   scan should be done in 2 levels:

   step 1: scan each individual block
   let the last thread of each block to 
   store its value to a global_array
   block_sum[blockIdx.x]

   step 2: scan the block_sum array

   step 3: add to every element its block offset
   found in block_sum array
   for example output[gid] = input[gid] + block_sum[blockIdx.x]


   */

__global__ void simple_scan(const unsigned int * const input,
                            unsigned int * const output,
                            const size_t n)
{
   __shared__ extern unsigned int data[];

   size_t gid = blockIdx.x * 2 * blockDim.x + threadIdx.x;
   size_t tid = 2 * threadIdx.x;
   size_t offset = 1;


   data[tid] = (gid < n) ? input[gid] : 0;
   data[tid+1] = (gid + 1 < n) ? input[gid+1] : 0;
   
   for (unsigned int d>>1; d > 0; d>>=1) {
      __syncthreads();

      if(threadIdx.x < d){
         int ai = offset * (tid +1) - 1;
         int bi = offset * (tid +2) - 1;
         data[bi] += data[ai];
      }
      offset *=2;
   }
   
   if (tid == 0) 
      data[n -1] = 0;

    
   for (int d = 1; d < n; d*=2) {
      offset >>= 1;
      __syncthreads();

      
      if(threadIdx.x < d){
         int ai = offset * (tid +1) - 1;
         int bi = offset * (tid +2) - 1;
         unsigned int t = data[ai];
         data[ai] = data[bi];
         data[bi] += t;
      }
   }

   __syncthreads();

   if(gid < n)
      output[gid] = data[tid];
    
    if(gid +1 < n)
      output[gid+1] = data[tid+1];

}

__global__ void apply_predicate(const unsigned int * const input, 
                                 unsigned int * const predicate,
                                 const size_t numElems,
                                 const unsigned int sb)
{
   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
   //size_t id = threadIdx.x;

   if(gid > numElems) return;

   predicate[gid] = (input[gid] & sb) == 0;

}


__global__ void change_positions(const unsigned int* const inputVals,
                                 const unsigned int* const inputPos,
                                 unsigned int* const outputVals,
                                 unsigned int* const outputPos,
                                 const unsigned int* const scan0,
                                 const unsigned int* const scan1,
                                 const unsigned int* const predicate,
                                 const size_t numElems,
                                 const unsigned int* sum)
{
   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
   //size_t id = threadIdx.x;
   //size_t offset = sum[0];
   size_t offset = 0;
  
   if(gid > numElems) return;

   if(predicate[gid] == 1){
      outputPos[scan0[gid]] = inputPos[gid]; 
      outputVals[scan0[gid]] = inputVals[gid]; 
   }else if(predicate[gid] == 0){
      //if(scan1[gid] + offset > numElems)
      //   printf("This is bad\n");
      outputPos[scan1[gid] + offset] = inputPos[gid]; 
      outputVals[scan1[gid] + offset] = inputVals[gid];       
   }else{
      printf("Ooops\n");
   }

}

// finalize this reduction
__global__ void reduce_sum(const unsigned int* const input, 
                           unsigned int * sum,
                           const size_t numElems)
{


   __shared__ extern unsigned int data[];

   size_t gid = blockIdx.x * 2 * blockDim.x + threadIdx.x;
   size_t id = threadIdx.x;

   //if(gid +blockDim.x > numElems) return;
   data[id] = gid < numElems ? input[gid] : 0;
      
   if(gid + blockDim.x < numElems)
      data[id] += input[gid + blockDim.x];

   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s>>=1){
      
      if(id < s){
         data[id] += data[id + s];
      }
      __syncthreads();
   }

   if (id == 0){
      sum[blockIdx.x] = data[id];
      //printf("Local Partial sum is %u and global is %u\n",data[id], sum[blockIdx.x]);
   }


}


//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(unsigned int *d_Data,
                              unsigned int *d_Buffer)
{
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        buf = d_Buffer[blockIdx.x];
    }

    __syncthreads();

    uint data4 = d_Data[pos];
    data4 += buf;
    //data4.x += buf;
    //data4.y += buf;
    //data4.z += buf;
    //data4.w += buf;
    d_Data[pos] = data4;
}


__global__ void flip_array(unsigned int* const input, 
                           const size_t numElems)
{
   size_t gid = blockIdx.x * 2 * blockDim.x + threadIdx.x;
   //size_t id = threadIdx.x;
   if(gid > numElems) return;
   input[gid] = !input[gid];
}

void your_sort(unsigned int*  d_inputVals,
               unsigned int*  d_inputPos,
               unsigned int*  d_outputVals,
               unsigned int*  d_outputPos,
               const size_t numElems)
{ 
  
   // Serial Part
   unsigned int * h_scan0 = new unsigned int[numElems];
   unsigned int * h_sum = new unsigned int[numElems];

   printf("numElems = %lu\n", numElems);

   size_t threads = 1024;
   size_t blocks = (numElems + threads -1) / threads;

   unsigned int * d_predicate;
   unsigned int * d_scan0;
   unsigned int * d_scan1;
   unsigned int * d_sum1;

   checkCudaErrors(cudaMalloc((void **) &d_predicate,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_scan0,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_scan1,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_sum1,  sizeof(unsigned int) * numElems));

   checkCudaErrors(cudaMemset(d_sum1,  0, sizeof(unsigned int) * numElems));

   unsigned int sb = 1<<0;   
   size_t memoryBytes = threads * sizeof(unsigned int);
   //for (size_t i = 0; i < 32; ++i){
   while(/*sb < (1<<31)*/ sb < (1<<1) ){
      apply_predicate<<<blocks, threads>>>(d_inputVals, d_predicate, 
                                          numElems, sb);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      simple_scan<<<blocks, threads, memoryBytes>>>(d_predicate, d_scan0, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      /// Serial Part
      checkCudaErrors(cudaMemcpy(h_scan0, d_predicate, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
      h_sum[0] = 0;
      for (int i = 0; i < numElems; ++i)
      {
         h_sum[0] += h_scan0[i];

      }
      printf("Host sum = %u\n",h_sum[0]);
      /// End of Serial part

      size_t elems = numElems;
      //size_t bytes = threads * sizeof(unsigned int);
      size_t numBlocks = (elems + 2*threads-1)/(2*threads);
      reduce_sum<<<numBlocks, threads, memoryBytes>>>(d_predicate, d_sum1, elems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      while(numBlocks > 1){
         elems = numBlocks;
         numBlocks = (elems + 2*threads-1)/(2*threads);
         reduce_sum<<<numBlocks, threads, memoryBytes>>>(d_sum1, d_sum1, elems);
         cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      }
      
      // @d_sum1[0] should be the final sum

      // Serial Part
      checkCudaErrors(cudaMemcpy(h_sum, d_sum1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      printf("Device sum = %u\n", h_sum[0]);
      // End of serial part

      flip_array<<<blocks, threads>>>(d_predicate, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


      simple_scan<<<blocks, threads, memoryBytes>>>(d_predicate, d_scan1, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      change_positions<<<blocks, threads>>>(d_inputVals, d_inputPos,
                                            d_outputVals, d_outputPos,
                                            d_scan0, d_scan1, d_predicate, 
                                            numElems, d_sum1);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      std::swap(d_inputPos, d_outputPos);
      std::swap(d_inputVals, d_outputVals);
      sb = sb << 1;
      std::cout << "end of iteration no " << sb <<'\n';
   }
   //std::swap(d_inputPos, d_outputPos);
   //   std::swap(d_inputVals, d_outputVals);
   std::cout << "its over\n";

}
