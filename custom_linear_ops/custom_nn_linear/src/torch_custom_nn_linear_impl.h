#pragma once

#include <iostream>
#include <vector>
#include "torch_custom_nn_linear.h"
#include <cuda_runtime_api.h>


namespace CustomOpsNNLinear
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
        // align the cuda memory by assigning the starting point with the power of 2
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct ReOrderedByKeyState
	{
		size_t size; // the key-value pair needed to be sorted 
		float* keys_unsorted;
		float* keys;
		int* values_unsorted;
		int* values;
		char* list_sorting_space; // the starting position of the current sortor

		static ReOrderedByKeyState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
        // return the size of P element plus 128 (redundant memory for alignment)
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};