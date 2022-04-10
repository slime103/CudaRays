#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>  
//#include <Eigen/dense>

struct pixel
{
	float r;
	float g;
	float b;
};

__global__ void shadePixels(pixel* p)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	//pixels are stored colunm/block num by row/thread offset
	//-----------------------------
	//---------------x-------------
	//-----------------------------
	int offset = (y * blockDim.x) + x;

	float r = 1.f - (float)y / gridDim.x;
	float g = 1.f - (float)x / blockDim.x;
	float b = 0.25f;

	p = p + offset;

	p->r = r * 255;
	p->g = g * 255;
	p->b = b * 255;

}

int main()
{
	const int image_width = 1024;
	const int image_height = 878;

	//create the pixel array
	pixel* pixels;

	cudaMallocManaged(&pixels, sizeof(pixel) * image_width * image_height);

	auto start_p1 = std::chrono::system_clock::now();

	//run the kernel, <<< numOf Blocks, numOf threads per block >>>
	shadePixels <<<image_height, image_width>>> (pixels);

	cudaDeviceSynchronize();

	auto start_p2 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_p1 = start_p2 - start_p1;

	//save the output to a file
	std::ofstream image_of;
	image_of.open("out/img.ppm", std::ios::out);

	if (!image_of.is_open())
	{
		std::cout << "File does not exist";
	}

	image_of << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int i = 0; i < image_height; i++)
	{ 
		for (int k = 0; k < image_width; k++)
		{
			pixel* p = pixels + (i * image_width) + k;
			
			image_of << static_cast<int>(p->r) << ' ' << static_cast<int>(p->g) << ' ' << static_cast<int>(p->b) << ' ';
		}
		image_of << "\n";
	}

	auto start_p3 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_p2 = start_p3 - start_p2;

	std::cout << "GPU computation elapsed time: " << elapsed_p1.count() << "s\n";
	std::cout << "Image output elapsed time: " << elapsed_p2.count() << "s\n";

	//clean up
	image_of.close();
	cudaFree(pixels);
	cudaDeviceReset();

	return 0;
}