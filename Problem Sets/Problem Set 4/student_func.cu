//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

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

__global__ void simple_scan(const bool * const input,
                            size_t * const output,
                            const size_t numElems)
{

   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
   size_t id = threadIdx.x;

   if(id > numElems) return;
   __shared__ extern unsigned int data[];

   data[id] = input[id];
   
   for (size_t s = 1; s < blockDim.x; s*=2)
   {
      __synchthreads();
      if(id >= s){
         data[id] += data[id - s];
      }
   }

   output[gid] = data[id];

}

__global__ void apply_predicate(const unsigned int * const input, 
                                 bool * const predicate,
                                 const size_t numElems,
                                 const unsigned int sb)
{
   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
   size_t id = threadIdx.x;

   if(id > numElems) return;

   predicate[id] = (input[id] & sb) == 0;

}


__global__ void change_positions(const unsigned int* const inputVals,
                                 const unsigned int* const inputPos,
                                 unsigned int* const outputVals,
                                 unsigned int* const outputPos,
                                 const size_t* const scan,
                                 const bool* const predicate,
                                 const size_t numElems)
{
   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
   size_t id = threadIdx.x;

   if(id > numElems) return;

   if(predicate[id]){
      outputPos[scan[id]] = inputPos[id]; 
      outputVals[scan[id]] = inputVals[id]; 
   }
   // TODO
   // we need to know the number of elements
   // after the fisrt scan


}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  
   size_t threads = 1024;
   size_t blocks = (numElems + threads -1) / threads;

   bool * d_predicate;
   size_t * d_scan;

   checkCudaErrors(cudaMalloc(d_predicate,  sizeof(bool) * numElems));
   checkCudaErrors(cudaMalloc(d_predicate,  sizeof(size_t) * numElems));

   unsigned int sb = 1<<0;   
   size_t memoryBytes = threads * sizeof(unsigned int);
   //for (size_t i = 0; i < 32; ++i){
   while(sb < UINT_MAX){
      apply_predicate<<<blocks, threads>>>(d_inputVals, d_predicate, 
                                          numElems, sb);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      simple_scan<<<blocks, threads, memoryBytes>>>(d_predicate, d_scan, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      change_positions<<<blocks, threads>>>(d_inputVals, d_inputPos,
                                            d_outputVals, d_outputPos,
                                            d_scan, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


      flip_array<<<blocks, threads>>>(d_predicate, numElems);
      //apply_predicate<<<blocks, threads>>>(d_inputVals, d_predicate, numElems
      //                                     sb, 1);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      simple_scan<<<blocks, threads, memoryBytes>>>(d_predicate, d_scan, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      change_positions<<<blocks, threads>>>(d_inputVals, d_inputPos,
                                            d_outputVals, d_outputPos,
                                            d_scan, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      std::swap(d_inputPos, d_outputPos);
      std::swap(d_inputVals, d_outputVals);
      sb = sb << 1;
   }

}
