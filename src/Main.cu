#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <algorithm>
#include "GLM/glm/vec3.hpp"
#include "GLM/glm/vec4.hpp"
#include "GLM/glm/geometric.hpp"
//#include "GLM/glm/common.hpp"

struct Vector4i
{
	int x;
	int y;
	int z;
	int vn1;
};

struct TriangleMesh
{
	int numOfVerts = 0;
	int numOfTris = 0;
	glm::vec3* verts = nullptr;
	glm::vec3* vnorms = nullptr;
	Vector4i* tris = nullptr;
};

static void parseOBJFile(const std::string filePath, TriangleMesh* triMesh)
{
	std::ifstream stream(filePath);

	std::string line;

	int vert_iterator = 0;
	int tris_iterator = 0;
	int vnorm_iterator = 0;

	while (getline(stream, line, ' '))
	{
		if (line == "v") //v x y z
		{
			glm::vec3 vertex;
			getline(stream, line, ' ');
			vertex.x = std::stof(line);
			getline(stream, line, ' ');
			vertex.y = std::stof(line);
			getline(stream, line, '\n');
			vertex.z = std::stof(line);

			*(triMesh->verts + vert_iterator++) = vertex;
			continue;
		}
		if (line.compare("vn") == 0)
		{
			glm::vec3 vnorm;
			getline(stream, line, ' ');
			vnorm.x = std::stof(line);
			getline(stream, line, ' ');
			vnorm.y = std::stof(line);
			getline(stream, line, '\n');
			vnorm.z = std::stof(line);

			*(triMesh->vnorms + vnorm_iterator++) = vnorm;
			continue;
		}
		if (line == "f") //f  v1/vt1/vn1 ..
		{
			Vector4i face;
			getline(stream, line, '/');
			face.x = std::stoi(line);
			getline(stream, line, '/');
			getline(stream, line, ' ');
			face.vn1 = std::stoi(line);
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

struct sphere
{
	glm::vec3 location;
	float r;

	__device__ sphere(float x, float y, float z, float r)
	{
		location.x = x;
		location.y = y;
		location.z = z;
		this->r = r;
	};
};

struct pointLight
{
	glm::vec3 location;
	glm::vec3 color;

	float i; //intensity

	__device__ pointLight(float x, float y, float z, float r, float g, float b, float intensity)
	{
		location.x = x;
		location.y = y;
		location.z = z;
		color.x = r;
		color.y = g;
		color.z = b;
		i = intensity;
	};
};

struct material
{
	glm::vec3 diffuse;
	glm::vec3 ambient;
	glm::vec3 specular;
	float power;

	__device__ material(float r1, float g1, float b1, float r2, float g2, float b2, float r3, float g3, float b3, float p)
	{
		diffuse.r = r1;
		diffuse.g = g1;
		diffuse.b = b1;
		ambient.r = r2;
		ambient.g = g2;
		ambient.b = b2;
		specular.r = r3;
		specular.g = g3;
		specular.b = b3;
		power = p;
	};
};
/*
__device__ inline float raySphere(glm::vec3 V, glm::vec3 W, sphere S) {
	V = V - S.location + W * 0.001f;
	float b = dot(V, W);
	float d = b * b - dot(V, V) + S.r * S.r;
	return d < 0. ? -1. : -b - sqrt(d);
}*/

__device__ inline float rayHalfspace(glm::vec3 V, glm::vec3 W, glm::vec4 H) {
	glm::vec4 V1 = glm::vec4(V, 1.);
	glm::vec4 W0 = glm::vec4(W, 0.);
	return -dot(V1, H) / dot(W0, H);
}

/* Return a point P if V, W intersects triangle ABC */
__device__ inline bool triangleTest(glm::vec3 V, glm::vec3 W, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 N, glm::vec3 &P)
{
	float d = -dot(N, a);
	glm::vec4 H = glm::vec4(N.x, N.y, N.z, d);

	//find t
	float t = rayHalfspace(V, W, H);

	//if t is positive we hit the plane infront of the camera
	if (t > 0.f && t < 1000.f)
	{
		//calculate point p
		glm::vec3 p = V + t * W;

		//make sure p is within the edges
		glm::vec3 ab = b - a;
		glm::vec3 ap = p - a;
		if (dot(cross(ab, ap), N) < 0.f) // > 0 means inside, = 0 means on the edge
		{
			return false;
		}

		glm::vec3 bc = c - b;
		glm::vec3 bp = p - b;
		if (dot(cross(bc, bp), N) < 0.f)
		{
			return false;
		}

		glm::vec3 ca = a - c;
		glm::vec3 cp = p - c;
		if (dot(cross(ca, cp), N) < 0.f)
		{
			return false;
		}

		//TODO calculate the barycentric coordinates to apply smooth shading with vertex normals

		//set values
		P = p;
		return true;
	}
	return false;
}

/*Returns a point P of the closest triangle*/
__device__ inline bool testTriangles(glm::vec3 V, glm::vec3 W, glm::vec3 &P, glm::vec3 &N, TriangleMesh* trimesh)
{
	bool hit = false;
	float z = 1000.f;

	//loop through all triangles in the scene
	for (int i = 0; i < trimesh->numOfTris; i++)
	{
		//vertex indexes begin at 1 for obj files, thus the -1
		if (triangleTest(V, W, *(trimesh->verts + (trimesh->tris + i)->x - 1), *(trimesh->verts + (trimesh->tris + i)->y - 1),
			*(trimesh->verts + (trimesh->tris + i)->z - 1), *(trimesh->vnorms + (trimesh->tris + i)->vn1 - 1), P))
		{ //Did we hit something?
			//is this point closer to the camera? 
			float dist = glm::distance(V, P);
			if (dist < z)
			{
				N = *(trimesh->vnorms + (trimesh->tris + i)->vn1 - 1);

				//we are closer so update the z position
				z = dist;
			}
			hit = true;
		}
	}

	//shade the closest point if we hit something
	return hit;
}

__device__ glm::vec3 shadePoint(glm::vec3 P, glm::vec3 W, glm::vec3 N, TriangleMesh* trimesh, material* m, pointLight* lights, int num_lights)
{
	glm::vec3 c = glm::vec3(0.f);
	for (int l = 0; l < num_lights; l++)
	{
		glm::vec3 contribution = glm::vec3(0.f);

		//calculate the light direction from the point lamps
		glm::vec3 Ld = (lights + l)->location - P;
		float distance = length(Ld);
		distance = distance * distance;
		Ld = normalize(Ld);

		// TODO: shadows for all surfaces

		//shadows from other triangles
		if (testTriangles((lights + l)->location, Ld, P, glm::vec3(0.f), trimesh))
		{
			continue;
		}

		glm::vec3 Rd = 2.f * N * dot(N, Ld) - Ld;
		contribution += (lights + l)->color * m->diffuse * glm::max(0.f, dot(N, Ld));
		contribution += m->specular * pow(glm::max(0.f, dot(Rd, -W)), m->power);
		contribution *= (lights + l)->i / distance;
		c += contribution;
	}

	return m->ambient + c;
	//return glm::vec3(1.f);
}

/*
__device__ inline glm::vec3 raySpheres(float tMin, glm::vec3 V, glm::vec3 W, sphere* spheres, int num_spheres, glm::vec3 color, material m, pointLight* lights, int num_lights)
{
	for (int n = 0; n < num_spheres; n++)
	{
		float t = raySphere(V, W, *(spheres + n));
		if (t > 0. && t < tMin) {
			glm::vec3 P = V + (W * t);
			glm::vec3 N = normalize(P - (spheres + n)->location);
			color = shadeSurface(P, W, N, spheres, num_spheres, m, lights, num_lights);
			tMin = t;
		}
	}
	return color;
}*/

__global__ void shadePixels(glm::vec3* p, TriangleMesh* trimesh)
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
	const float image_plane_height = 0.06f; // 6cm

	//calculate pixel area to sample over (coordinates begin at the top left of each pixel)
	//const float pixel_width = image_plane_width / blockDim.x;

	//this is where pixel sample points should be generated, before conversion to camera space

	//find this pixel's camera space position
	glm::vec3 pixelpos;
	pixelpos.x = (float) pixel_x / blockDim.x;
	pixelpos.x = pixelpos.x * image_plane_width - (image_plane_width / 2.0f);
	pixelpos.y = (float) pixel_y / gridDim.x;
	pixelpos.y = -pixelpos.y * image_plane_height + (image_plane_height / 2.0f);
	pixelpos.z = 0.f;

	//select a focal length, aperature, and focus distance
	const float focal_length = 0.05f; // 50mm
	//const float aperture = 16.0f; // f16
	//const float focus_dist = 3.f; // 3m

	//Set V and W ... TODO add lens configurations
	glm::vec3 V = glm::vec3(0.f, 0.f, focal_length);
	glm::vec3 W = glm::vec3(V.x + pixelpos.x,  V.y + pixelpos.y, -1.f * V.z);

	//set the background color
	glm::vec3 color = glm::vec3(0.35f, 0.35f, 0.4f);

	// TEMP DEFINE SPHERE DATA
	//sphere spheres[3] = {sphere(0.f, 0.f, -5.f, 1.f), sphere(0.5f, 0.f, -3.f, 1.f), sphere(-0.5f, 0.f, -7.f, 1.f)};
	// TEMP DEFINE LIGHT DATA
	const int num_lights = 2;
	pointLight lights[num_lights] = { pointLight(0.5f, 3.5f, -5.5f, .59f, .93f, .59f, 25.f), pointLight(1.5f, -.5f, -1.5f, .19f, .63f, .49f, 15.f) };
	// TEMP DEFINE MATERIAL DATA
	material sphere_mat = material(0.1f, 0.1f, 0.1f, 0.2f, 0.4f, 0.5f, 1.f, 1.f, 1.f, 1.5f);

	//color = raySpheres(-1.f, V, W, &spheres[0], 3, color, sphere_mat, &lights[0], 1);

	glm::vec3 P;
	glm::vec3 N;

	if (testTriangles(V, W, P, N, trimesh))
	{
		color = shadePoint(P, W, N, trimesh, &sphere_mat, &lights[0], num_lights);
	}
	
	//color += triangleTest(V, W, *(trimesh->verts + (trimesh->tris)->x), *(trimesh->verts + (trimesh->tris)->y),
		//*(trimesh->verts + (trimesh->tris)->z), &sphere_mat, &light, 1);
	//color += triangleTest(V, W, glm::vec3(-0.1f, 0.0f, -1.f), glm::vec3(0.1f, 0.0f, -1.f), glm::vec3(0.f, 0.1f, -1.f), glm::vec3(0.f, 0.f, 1.f), &sphere_mat, &light, 1);

	//Set the final color
	p->r = glm::clamp(color.x * 255.f, 0.f, 255.f);
	p->g = glm::clamp(color.y * 255.f, 0.f, 255.f);
	p->b = glm::clamp(color.z * 255.f, 0.f, 255.f);

}

int main()
{
	const int image_width = 1024;
	const int image_height = 878;

	//create the pixel array
	glm::vec3* pixels;

	cudaMallocManaged(&pixels, sizeof(glm::vec3) * image_width * image_height);

	
	//Read in Triangle information from OBJ file
	TriangleMesh* trimesh;
	glm::vec3* verts;
	glm::vec3* vnorms;
	Vector4i* tris;
	cudaMallocManaged(&trimesh, sizeof(TriangleMesh));
	cudaMallocManaged(&verts, sizeof(glm::vec3) * 1000);
	cudaMallocManaged(&vnorms, sizeof(glm::vec3) * 1000);
	cudaMallocManaged(&tris, sizeof(Vector4i) * 1000);
	trimesh->verts = verts;
	trimesh->vnorms = vnorms;
	trimesh->tris = tris;

	auto start_p0 = std::chrono::system_clock::now();

	parseOBJFile("in/mnky.obj", trimesh);

	auto start_p1 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_p0 = start_p0 - start_p1;

	//run the kernel, <<< numOf Blocks, numOf threads per block >>>
	shadePixels <<<image_height, image_width>>> (pixels, trimesh);

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
			glm::vec3* p = pixels + (i * image_width) + k;
			
			image_of << static_cast<int>(p->r) << ' ' << static_cast<int>(p->g) << ' ' << static_cast<int>(p->b) << ' ';
		}
		image_of << "\n";
	}

	auto start_p3 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_p2 = start_p3 - start_p2;

	std::cout << "Obj input elapsed time: " << elapsed_p0.count() << "s\n";
	std::cout << "GPU computation elapsed time: " << elapsed_p1.count() << "s\n";
	std::cout << "Image output elapsed time: " << elapsed_p2.count() << "s\n";

	//clean up
	image_of.close();
	cudaFree(trimesh->verts);
	cudaFree(trimesh->vnorms);
	cudaFree(trimesh->tris);
	cudaFree(trimesh);
	cudaFree(pixels);
	cudaDeviceReset();

	return 0;
}