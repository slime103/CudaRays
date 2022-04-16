#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
//#include <Eigen/dense>
#include "Vector3f.h"

struct Vector3i
{
	int x;
	int y;
	int z;
};

struct TriangleMesh
{
	int numOfVerts = 0;
	int numOfTris = 0;
	Vector3f* verts = nullptr;
	Vector3i* tris = nullptr;
};

static void parseOBJFile(const std::string filePath, TriangleMesh* triMesh)
{
	std::ifstream stream(filePath);

	std::string line;

	int vert_iterator = 0;
	int tris_iterator = 0;

	while (getline(stream, line, ' '))
	{
		if (line == "v") //v x y z
		{
			Vector3f vertex;
			getline(stream, line, ' ');
			vertex.x = std::stof(line);
			getline(stream, line, ' ');
			vertex.y = std::stof(line);
			getline(stream, line, '\n');
			vertex.z = std::stof(line);

			*(triMesh->verts + vert_iterator++) = vertex;
			continue;
		}
		if (line == "f") //f  v1/vt1/vn1 ..
		{
			Vector3i face;
			getline(stream, line, '/');
			face.x = std::stoi(line);
			getline(stream, line, ' ');
			getline(stream, line, '/');
			face.y = std::stoi(line);
			getline(stream, line, ' ');
			getline(stream, line, '/');
			face.z = std::stoi(line);

			*(triMesh->tris + tris_iterator++) = face;
		}
		getline(stream, line, '\n');
	}

	triMesh->numOfVerts = vert_iterator;
	triMesh->numOfTris = tris_iterator;


	stream.close();
}

struct pixel
{
	float r;
	float g;
	float b;
};

__global__ void shadePixels(pixel* p)
{
	int pixel_x = threadIdx.x;
	int pixel_y = blockIdx.x;

	//pixels are stored colunm/block num by row/thread offset
	//-----------------------------
	//---------------x-------------
	//-----------------------------
	p = p + (pixel_y * blockDim.x) + pixel_x;

	//set the film size
	const float image_plane_width = 0.07f; // 7cm
	const float image_plane_height = 0.06f; //6cm

	//calculate pixel area to sample over (coordinates begin at the top left of each pixel)
	const float pixel_width = image_plane_width / blockDim.x;

	//this is where pixel sample points should be generated, before conversion to camera space

	//find this pixel's camera space position
	Vector3f pixelpos;
	pixelpos.x = (float) pixel_x / blockDim.x;
	pixelpos.x = pixelpos.x * image_plane_width - (image_plane_width / 2.0f);
	pixelpos.y = (float) pixel_y / gridDim.x;
	pixelpos.y = -pixelpos.y * image_plane_height + (image_plane_height / 2.0f);
	pixelpos.z = 0.f;

	//select a focal length, aperature, and focus distance
	const float focal_length = 0.05f;
	const float aperture = 16.0f;
	const float focus_dist = 3.f;

	//Set V and W ... TODO add lens configurations
	Vector3f V = Vector3f(0.f, 0.f, focal_length);
	Vector3f W = Vector3f(V.x + pixelpos.x,  V.y + pixelpos.y, -1.f * V.z);

	//set the background color
	Vector3f color = Vector3f(0.2f, 0.3f, 0.4f);

	// TEMP DEFINE SPHERE DATA
	sph

	//Set the final color
	p->r *= 255;
	p->g *= 255;
	p->b *= 255;

}

int main()
{
	const int image_width = 1024;
	const int image_height = 878;

	//create the pixel array
	pixel* pixels;

	cudaMallocManaged(&pixels, sizeof(pixel) * image_width * image_height);

	/*
	//Read in Triangle information from OBJ file
	TriangleMesh* trimesh;
	cudaMallocManaged(&trimesh, sizeof(TriangleMesh));
	cudaMallocManaged(&trimesh->verts, sizeof(Vector3f) * 1000);
	cudaMallocManaged(&trimesh->tris, sizeof(Vector3i) * 1000);

	parseOBJFile("in/mnky.obj", trimesh);
	*/

	//Create lights, for now a light is a point

	//Create spheres, a point and a radius

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
	cudaFree(trimesh->verts);
	cudaFree(trimesh->tris);
	cudaFree(trimesh);
	cudaFree(pixels);
	cudaDeviceReset();

	return 0;
}