#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <cmath>		/* log2() */
#include <cstdint> 		/* int64_t, uint64_t */
#include <cstdlib>		/* srand(), rand() */
#include <ctime>		/* time() */
#include <iostream> 		/* std::cout, std::endl */
#include <time.h>
#include <stdlib.h>
#include "utils.h" 	//INCLUDE HEADER FILE
#include "ntt.h"

using namespace std;
/**
 * Return vector with each element of the input at its bit-reversed position
 *
 * @param vec The vector to bit reverse
 * @param n   The length of the vector, must be a power of two
 * @return    The bit reversed vector
 */
__host__ __device__ uint64_t* bit_reverse(uint64_t* vec, uint64_t n) {

	uint64_t num_bits = log2(n);

	uint64_t* result;
	result = (uint64_t*)malloc(n * sizeof(uint64_t));

	uint64_t reverse_num;
	for (uint64_t i = 0; i < n; i++) {

		reverse_num = 0;
		for (uint64_t j = 0; j < num_bits; j++) {

			reverse_num = reverse_num << 1;
			if (i & (1 << j)) {
				reverse_num = reverse_num | 1;
			}
		}

		result[reverse_num] = vec[i];

	}

	return result;
}

/**
 * Compare two vectors element-wise and return whether they are equivalent
 *
 * @param vec1	The first vector to compare
 * @param vec2 	The second vector to compare
 * @param n 	The length of the vectors
 * @param debug	Whether to print debug information (will run entire vector)
 * @return 	Whether the two vectors are element-wise equivalent
 */
bool compVec(uint64_t* vec1, uint64_t* vec2, uint64_t n, bool debug) {

	bool comp = true;
	for (uint64_t i = 0; i < n; i++) {

		if (vec1[i] != vec2[i]) {
			comp = false;

			if (debug) {
				std::cout << "(vec1[" << i << "] : " << vec1[i] << ")";
				std::cout << "!= (vec2[" << i << "] : " << vec2[i] << ")";
				std::cout << std::endl;
			}
			else {
				break;
			}
		}
	}

	return comp;
}

/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__host__ __device__ uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m) {

	uint64_t result = 1;

	while (exp > 0) {

		if (exp % 2) {

			result = modulo(result * base, m);

		}

		exp = exp >> 1;
		base = modulo(base * base, m);
	}

	return result;
}

/**
 * Perform the operation 'base (mod m)'
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__host__ __device__ uint64_t modulo(int64_t base, int64_t m) {
	int64_t result = base % m;

	return (result >= 0) ? result : result + m;
}

/**
 * Print an array of arbitrary length in a readable format
 *
 * @param vec	The array to be displayed
 * @param n	The length of the array
 */

 //void printVec(uint64_t* vec, uint64_t n) {
 //
 //	std::cout << "[";
 //
 //	for (uint64_t i = 0; i < n; i++) {
 //
 //		std::cout << vec[i] << ",";
 //
 //	}
 //
 //	std::cout << "]" << std::endl;
 //}

 /**
  * Generate an array of arbitrary length containing random positive integers
  *
  * @param n	The length of the array
  * @param max	The maximum value for an array element [Default: RAND_MAX]
  */
__host__ __device__ uint64_t* randVec(uint64_t n, uint64_t max) {

	uint64_t* vec;

	vec = (uint64_t*)malloc(n * sizeof(uint64_t));

	srand(time(0));
	for (uint64_t i = 0; i < n; i++) {

		vec[i] = rand() % (max + 1);

	}

	return vec;
}

//NTT.cpp Code

/**
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 */
__global__ void kernelNTT_DIT(uint64_t* result, uint64_t* vec, uint64_t n, uint64_t p, uint64_t r, uint64_t* mpow, uint64_t* akp, uint64_t logged, bool rev) {
	/*int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;*/

	uint64_t m, k_, a, factor1, factor2;
	//if (rev) {
	//	result = bit_reverse(vec, n);
	//}
	//else {
	//	for (uint64_t i = 0; i < n; i++) {
	//		result[i] = vec[i];
	//	}
	//}


	for (uint64_t i = 1; i <= logged; i++) {

		m = mpow[i - 1];

		k_ = (p - 1) / m;
		a = modExp(r, k_, p);

		for (uint64_t j = 0; j < n; j += m) {

			for (uint64_t k = 0; k < m / 2; k++) {

				factor1 = result[j + k];
				factor2 = modulo(akp[(i-1)*m/2+k] * result[j + k + m / 2], p);

				result[j + k] = modulo(factor1 + factor2, p);
				result[j + k + m / 2] = modulo(factor1 - factor2, p);


			}
		}

	}
	return;
}

void cudahelp(uint64_t* result, uint64_t* vec, uint64_t n, uint64_t p, uint64_t r, uint64_t* mpow, uint64_t* akp, int logged, bool rev)
{


	uint64_t* dev_a = nullptr;
	uint64_t* dev_c = nullptr;
	uint64_t* dev_mpow = nullptr;
	uint64_t* dev_akp = nullptr;

	cudaMalloc((void**)&dev_mpow, logged * sizeof(uint64_t));
	cudaMalloc((void**)&dev_c, n * sizeof(uint64_t));
	cudaMalloc((void**)&dev_a, n * sizeof(uint64_t));
	cudaMalloc((void**)&dev_akp, 12*2048 * sizeof(uint64_t));

	cudaMemcpy(dev_mpow, mpow, logged * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, vec, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, result, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_akp, akp, 12*2048 * sizeof(uint64_t), cudaMemcpyHostToDevice);


	//dim3 block(32,32);
	kernelNTT_DIT << <1, 1 >> > (dev_c, dev_a, n, p, r, dev_mpow,dev_akp, logged, false);
	cudaDeviceSynchronize();

	cudaMemcpy(result, dev_c, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_mpow);

}

//MAIN.cpp



int main(int argc, char* argv[]) {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();

	const uint64_t n = 4096;
	uint64_t p = 68719403009;
	uint64_t r = 36048964756;

	const int logged = log2(n);

	uint64_t vec[n];
	uint64_t* result;

	for (int i = 0; i < n; i++) {
		vec[i] = i;
	}

	uint64_t mpow[12];
	uint64_t k_, a;
	uint64_t akp[12*2048];

	for (uint64_t i = 1; i <= 12; i++) {

		mpow[i - 1] = pow(2, i);
		k_ = (p - 1) / mpow[i - 1];
		a = modExp(r, k_, p);
		for (uint64_t k = 0; k < mpow[i - 1] / 2; k++) {
			akp[(i-1)* mpow[i - 1]/2 +k] = modExp(a, k, p);
		}
	}
	result = bit_reverse(vec, n);
	cudahelp(result, vec, n, p, r, mpow,akp, logged, false);

	auto t2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << "[";
	for (uint64_t i = 0; i < n; i++) {

		std::cout << result[i] << ",";

	}
	std::cout << "]" << std::endl;

	std::cout << ms_double.count() << "ms";
	return 0;
}
