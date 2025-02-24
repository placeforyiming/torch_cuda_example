#pragma once

#include <iostream>
#include <vector>
#include "torch_template.h"
#include <cuda_runtime_api.h>


namespace TorchOpsTemplate
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
        // align the cuda memory by assigning the starting point with the power of 2
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct InclusiveSumState
	{
		size_t scan_size;
		char* scanning_space;
		uint32_t* list_inclusive_sumed;
		uint32_t* list;

		static InclusiveSumState fromChunk(char*& chunk, size_t P);
	};


	struct ReOrderedByKeyState
	{
		size_t size;
		uint32_t* keys_unsorted;
		uint32_t* keys;
		uint32_t* list_unsorted;
		uint32_t* list;
		char* list_sorting_space;

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