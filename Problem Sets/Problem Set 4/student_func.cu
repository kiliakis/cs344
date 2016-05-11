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

__global__ void simple_scan(const unsigned int * const input,
                            unsigned int * const output,
                            const size_t numElems)
{

   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
   size_t id = threadIdx.x;

   if(id > numElems) return;
   __shared__ extern unsigned int data[];

   data[id] = input[id];
   
   for (unsigned int s = 1; s < blockDim.x; s*=2)
   {
      __syncthreads();
      if(id >= s){
         data[id] += data[id - s];
      }
   }

   output[gid] = data[id];

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
   }else{
      //if(scan1[gid] + offset > numElems)
      //   printf("This is bad\n");
      outputPos[scan1[gid] + offset] = inputPos[gid]; 
      outputVals[scan1[gid] + offset] = inputVals[gid];       
   }

}

// finalize this reduction
__global__ void reduce_sum(const unsigned int* const input, 
                           unsigned int * const sum,
                           const size_t numElems)
{
   size_t gid = blockIdx.x * 2 * blockDim.x + threadIdx.x;
   size_t id = threadIdx.x;

   if(gid +blockDim.x > numElems) return;
   __shared__ extern unsigned int data[];

   data[id] = input[gid] + input[gid + blockDim.x];

   for (unsigned int s = blockDim.x/2; s > 0; s>>=1){
      __syncthreads();
      unsigned int other = id + s;
      if(id < s && other < numElems){
         data[id] += data[other];
      }
   }

   if (id == 0){
      sum[blockIdx.x] = data[id];
   }


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
  
   size_t threads = 1024;
   size_t blocks = (numElems + threads -1) / threads;

   unsigned int * d_predicate;
   unsigned int * d_scan0;
   unsigned int * d_scan1;
   unsigned int * d_sum1;
   unsigned int * d_sum2;

   checkCudaErrors(cudaMalloc((void **) &d_predicate,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_scan0,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_scan1,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_sum1,  sizeof(unsigned int) * numElems));
   checkCudaErrors(cudaMalloc((void **) &d_sum2,  sizeof(unsigned int) * numElems));

   unsigned int sb = 1<<0;   
   size_t memoryBytes = threads * sizeof(unsigned int);
   //for (size_t i = 0; i < 32; ++i){
   while(sb < (1<<31)){
      apply_predicate<<<blocks, threads>>>(d_inputVals, d_predicate, 
                                          numElems, sb);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      simple_scan<<<blocks, threads, memoryBytes>>>(d_predicate, d_scan0, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

     
      // This should be working
      size_t elems = numElems;
      checkCudaErrors(cudaMemcpy(d_predicate, d_sum1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
      do{
         size_t numBlocks = (elems + threads-1)/threads;
         size_t bytes = threads * sizeof(unsigned int);
         reduce_sum<<<numBlocks, threads, bytes>>>(d_sum1, d_sum2, elems);
         elems = numBlocks;
         std::swap(d_sum1, d_sum2);
      }while(elems > 1);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      // @d_sum2[0] should be the final sum

      flip_array<<<blocks, threads>>>(d_predicate, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


      simple_scan<<<blocks, threads, memoryBytes>>>(d_predicate, d_scan1, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      change_positions<<<blocks, threads>>>(d_inputVals, d_inputPos,
                                            d_outputVals, d_outputPos,
                                            d_scan0, d_scan1, d_predicate, 
                                            numElems, d_sum2);
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
