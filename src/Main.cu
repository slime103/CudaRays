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
#include <curand.h>
#include <curand_kernel.h>
//#include "SimplexNoise.h"

struct face
{
	int x;
	int y;
	int z;
	int vn1;
	int vn2;
	int vn3;
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

struct VertexData
{
	int numOfVerts = 0;
	glm::vec3* verts = nullptr;
	glm::vec3* vnorms = nullptr;
	//TODO: add vertex textures
};

struct TriangleMesh
{
	char name[10];
	material* mat;
	int numOfTris = 0;
	face* tris = nullptr;
};

static void parseOBJFile(const std::string filePath, VertexData* vertData, TriangleMesh* triMeshes, int maxMeshes, int &numOfMeshes)
{
	std::ifstream stream(filePath);

	std::string line;

	int mesh_iterator = -1; //gets set to 0 on first obj scan
	int vert_iterator = 0;
	int tris_iterator = 0;
	int vnorm_iterator = 0;
	//int vert_culmlative = 0;
	//int vn_culmlative = 0;

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

			*(vertData->verts + vert_iterator++) = vertex;
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

			*(vertData->vnorms + vnorm_iterator++) = vnorm;
			continue;
		}
		if (line == "f") //f  v1/vt1/vn1 ..
		{
			face face;
			getline(stream, line, '/');
			face.x = std::stoi(line);
			getline(stream, line, '/');
			getline(stream, line, ' ');
			face.vn1 = std::stoi(line);
			getline(stream, line, '/');
			face.y = std::stoi(line);
			getline(stream, line, '/');
			getline(stream, line, ' ');
			face.vn2 = std::stoi(line);
			getline(stream, line, '/');
			face.z = std::stoi(line);
			getline(stream, line, '/');
			getline(stream, line, '\n');
			face.vn3 = std::stoi(line);

			*((triMeshes + mesh_iterator)->tris + tris_iterator++) = face;
			(triMeshes + mesh_iterator)->numOfTris = tris_iterator;
			continue;
		}
		if (line == "o") //object name
		{
			if (mesh_iterator == maxMeshes-1)
			{
				break; //max number of meshes reached
			}

			//prepare scanning for next mesh
			mesh_iterator++;
			tris_iterator = 0;

			getline(stream, line, '\n');
			std::strcpy((triMeshes + mesh_iterator)->name, line.c_str());		

			continue;
		}
		getline(stream, line, '\n');
	}
	vertData->numOfVerts = vert_iterator;
	numOfMeshes = mesh_iterator+1;
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
__device__ inline bool testTriangles(glm::vec3 V, glm::vec3 W, glm::vec3 &P, glm::vec3 &N, VertexData* vertData, TriangleMesh* trimesh, int numOfMeshes)
{
	bool hit = false;
	float z = 1000.f;

	//loop through all triangles in the scene
	for (int k = 0; k < numOfMeshes; k++)
	{
		for (int i = 0; i < trimesh->numOfTris; i++)
		{
			//vertex indexes begin at 1 for obj files, thus the -1
			if (triangleTest(V, W, *(vertData->verts + ((trimesh + k)->tris + i)->x - 1), *(vertData->verts + ((trimesh + k)->tris + i)->y - 1),
				*(vertData->verts + ((trimesh + k)->tris + i)->z - 1), *(vertData->vnorms + ((trimesh + k)->tris + i)->vn1 - 1), P))
			{ //Did we hit something?
				//is this point closer to the camera? 
				float dist = glm::distance(V, P);
				if (dist < z)
				{
					N = *(vertData->vnorms + ((trimesh + k)->tris + i)->vn1 - 1);

					//we are closer so update the z position
					z = dist;
				}
				hit = true;
			}
		}
	}

	//shade the closest point if we hit something
	return hit;
}

/*Returns true if any triangle is hit*/
__device__ inline bool testTrianglesAny(glm::vec3 V, glm::vec3 W, VertexData* vertData, TriangleMesh* trimesh, int numOfMeshes)
{
	glm::vec3 P;

	//loop through all triangles in the scene
	for (int k = 0; k < numOfMeshes; k++)
	{
		for (int i = 0; i < trimesh->numOfTris; i++)
		{
			//vertex indexes begin at 1 for obj files, thus the -1
			if (triangleTest(V, W, *(vertData->verts + ((trimesh + k)->tris + i)->x - 1), *(vertData->verts + ((trimesh + k)->tris + i)->y - 1),
				*(vertData->verts + ((trimesh + k)->tris + i)->z - 1), *(vertData->vnorms + ((trimesh + k)->tris + i)->vn1 - 1), P))
			{ //Did we hit something?
				return true;
			}
		}
	}

	return false;
}

__device__ glm::vec3 shadePoint(glm::vec3 P, glm::vec3 W, glm::vec3 N, VertexData* vertData, TriangleMesh* trimesh, material* m, pointLight* lights, int num_lights, int numOfMeshes)
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
		//trace from the point back to the light, is another triangle in the way?
		if (testTrianglesAny(P, Ld, vertData, trimesh, numOfMeshes))
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

struct Camera
{
	//set the film size
	const float image_plane_width = 0.07f; // 7cm
	const float image_plane_height = 0.06f; // 6cm

	//select a focal length, aperature, and focus distance
	const float focal_length = 0.05f; // 50mm
	const float aperture = 16.0f; // f16
	const float focus_dist = 3.f; // 3m
};

//Returns a point on a concentric disk, takes x and y values between -1 and 1
__device__ inline void sampleDisk(float &x, float &y)
{
	static const float PiOver2 = 1.570796326794896619231321691639;
	static const float PiOver4 = 0.785398163397448309615660845819;

	float theta, r;

	if (abs(x) > abs(y)) 
	{
		r = x;
		theta = PiOver4 * (y / x);
	}
	else 
	{
		r = y;
		theta = PiOver2 - PiOver4 * (x / y);
	}

	x = r * cos(theta);
	y = r * sin(theta);
}
/*
//Takes a ray, V, W, and returns a new ray direction using a lens sample point
__device__ inline void dofRay(glm::vec3 &V, glm::vec3 &W, const Camera& camera)
{
	//Sample a point on a disk, SP (sample point)
	float x, y, z;

	x = SimplexNoise::noise((float) blockDim.x * blockIdx.x + threadIdx.x);
	y = SimplexNoise::noise((float) gridDim.x * blockIdx.x + threadIdx.x);
	z = SimplexNoise::noise(gridDim.x + (float) blockIdx.x * threadIdx.x);

	if (z > -0.5f)
	{
		x = -x;
	}
	else if (z > 0.f)
	{
		y = -y;
	}
	else if (z > 0.5f)
	{
		x = -x;
		y = -y;
	}

	sampleDisk(x, y);

	//Convert unit disk coordinates to lens diameter, LP (lens point)
	V.x = x * camera.focal_length / camera.aperture;
	V.y = y * camera.focal_length / camera.aperture;

	//calculate the focal point by scaling the primary ray and adding it to the ray origin position
	glm::vec3 focalPoint = V + (W * camera.focus_dist);

	//Construct a unit vector from the sample point to the focal point, FD (focal direction)
	W = normalize(focalPoint - V);
}*/

__device__ inline glm::vec3 shadePoint(const Camera &camera, const float pixel_x, const float pixel_y, VertexData* vertData, TriangleMesh* trimesh, int numOfMeshes)
{
	//find this pixel's camera space position
	glm::vec3 pixelpos;
	pixelpos.x = pixel_x / blockDim.x;
	pixelpos.x = pixelpos.x * camera.image_plane_width - (camera.image_plane_width / 2.0f);
	pixelpos.y = pixel_y / gridDim.x;
	pixelpos.y = -pixelpos.y * camera.image_plane_height + (camera.image_plane_height / 2.0f);
	pixelpos.z = 0.f;

	//Set V and W ... TODO add lens configurations
	glm::vec3 V = glm::vec3(0.f, 0.f, camera.focal_length);
	glm::vec3 W = glm::vec3(V.x + pixelpos.x, V.y + pixelpos.y, -1.f * V.z);

	//dofRay(V, W, camera);

	//set the background color
	glm::vec3 color = glm::vec3(0.6f, 0.8f, 0.8f);

	// TEMP DEFINE LIGHT DATA
	const int num_lights = 2;
	pointLight lights[num_lights] = { pointLight(0.5f, 5.5f, -5.5f, .59f, .93f, .59f, 25.f), pointLight(1.5f, -.5f, -1.5f, .19f, .63f, .49f, 3.f) };
	// TEMP DEFINE MATERIAL DATA
	material sphere_mat = material(0.1f, 0.1f, 0.1f, 0.2f, 0.4f, 0.5f, 1.f, 1.f, 1.f, 1.5f);

	glm::vec3 P;
	glm::vec3 N;

	if (testTriangles(V, W, P, N, vertData, trimesh, numOfMeshes))
	{
		color = shadePoint(P, W, N, vertData, trimesh, &sphere_mat, &lights[0], num_lights, numOfMeshes);
	}

	return color;
}

__global__ void shadePixels(glm::vec3* pixelptr, VertexData* vertData, TriangleMesh* trimesh, int numOfMeshes)
{
	int pixel_x = threadIdx.x;
	int pixel_y = blockIdx.x;

	//pixels are stored colunm/block num by row/thread offset
	//-----------------------------
	//---------------x-------------
	//-----------------------------
	pixelptr = pixelptr + (pixel_y * blockDim.x) + pixel_x;

	Camera camera;

	const int num_pixel_samples = 4; //must be divisible by 2

	//curandState_t state;

	glm::vec3 color = glm::vec3(0.);

	for (int i = 0; i < num_pixel_samples; i++)
	{
		//TODO: divide the pixel further into cells
		/*for (int j = 0; j < num_pixel_samples; j++)
		{
			color += shadePoint(camera, pixel_x + (1 / num_pixel_samples) * i + (1 / (num_pixel_samples * 2)), 
			pixel_y + (1 / num_pixel_samples) * j + (1 / (num_pixel_samples * 2)), trimesh);
		}*/
		
		//this takes samples in an x pattern
		float rand_x, rand_y;
		if (i < num_pixel_samples/2)
		{
			rand_x = pixel_x + 1 / num_pixel_samples * i;
			rand_y = pixel_y + 1 / num_pixel_samples * i;
		}
		else
		{
			rand_x = (pixel_x + 1) - 1 / num_pixel_samples * i;
			rand_y = pixel_y + 1 / num_pixel_samples * i;
		}
		color += shadePoint(camera, rand_x,	rand_y, vertData, trimesh, numOfMeshes);

		//select a random point within the interval
		//curand_init(2127+i, (blockDim.x * pixel_y + pixel_x), 0, &state);
		//float rand_x = pixel_x + curand_uniform(&state) * pixel_width;
		//float rand_y = pixel_y + curand_uniform(&state) * pixel_height;
	}

	//average the pixel sample colors
	color /= num_pixel_samples;

	//Set the final color
	pixelptr->r = glm::clamp(color.x * 255.f, 0.f, 255.f);
	pixelptr->g = glm::clamp(color.y * 255.f, 0.f, 255.f);
	pixelptr->b = glm::clamp(color.z * 255.f, 0.f, 255.f);

}

int main()
{
	const int image_width = 1024;
	const int image_height = 878;

	//create the pixel array
	glm::vec3* pixels;

	cudaMallocManaged(&pixels, sizeof(glm::vec3) * image_width * image_height);

	
	//Read in Triangle information from OBJ file
	TriangleMesh* trimeshes;
	VertexData* vertData;
	const int maxMeshes = 3;
	const int maxVertices = 300; //For the entire scene
	const int maxFaces = 300; //Per object
	
	cudaMallocManaged(&trimeshes, sizeof(TriangleMesh) * maxMeshes);
	cudaMallocManaged(&vertData, sizeof(vertData));

	glm::vec3* verts;
	glm::vec3* vnorms;
	cudaMallocManaged(&verts, sizeof(glm::vec3) * maxVertices * maxMeshes);
	cudaMallocManaged(&vnorms, sizeof(glm::vec3) * maxVertices * maxMeshes);
	vertData->verts = verts;
	vertData->vnorms = vnorms;

	//Initialize the triangle arrays for all meshes
	for (int i = 0; i < maxMeshes; i++)
	{
		face* tris;
		
		cudaMallocManaged(&tris, sizeof(face) * maxFaces);
		
		(trimeshes + i)->tris = tris;
	}
	

	auto start_p0 = std::chrono::system_clock::now();

	int numOfMeshes; //we don't know how many there will be so save the result here
	parseOBJFile("in/mnky.obj", vertData, trimeshes, maxMeshes, numOfMeshes);

	auto start_p1 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_p0 = start_p0 - start_p1;

	//run the kernel, <<< numOf Blocks, numOf threads per block >>>
	shadePixels <<<image_height, image_width>>> (pixels, vertData, trimeshes, numOfMeshes);

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

	for (int i = 0; i < maxMeshes; i++)
	{
		
		cudaFree((trimeshes + i)->tris);
	}
	cudaFree(trimeshes);

	cudaFree(vertData->verts);
	cudaFree(vertData->vnorms);
	cudaFree(vertData);
	
	cudaFree(pixels);
	cudaDeviceReset();

	return 0;
}