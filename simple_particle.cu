#include "simple_particle.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"
#include <stdio.h>

__constant__ simpleParticleSystem d_sps[1];

__global__ void generateParticles();

__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height);

__global__ void updateParticles(float passed_time);

__device__ float2 get_normal_vector(float rand_num);

__device__ float get_energy(float2 p1, float2 p2, float dist_bound_powerd);

__device__ uchar4 get_color_from_energy(float energy);

__device__ float2 get_acceleration(int index);

__device__ void update_particle_velocity(int index, float2 acc);

__device__ int update_particle_possition(int index); //return whether the particle is dead

void init_particles_cuda(simpleParticleSystem &sps) {
	int max_num_particles = sps.MAX_PARTICLE_SIZE;
	int one_batch_num_particles = sps.ONE_BATCH_PARTICLE_SIZE;

	cudaMalloc((void**)&sps.energy, sizeof(*sps.energy)*max_num_particles);
	cudaMalloc((void**)&sps.position, sizeof(*sps.position)*max_num_particles);
	cudaMalloc((void**)&sps.velocity, sizeof(*sps.velocity)*max_num_particles);
	cudaMalloc((void**)&sps.remain_time, sizeof(*sps.remain_time)*max_num_particles);
	cudaMalloc((void**)&sps.rand_data, sizeof(*sps.rand_data)*one_batch_num_particles*3);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Memory Allocation Error: %s\n", cudaGetErrorString(err));
}

void destroy_particles_cuda(simpleParticleSystem &sps) {
	cudaError_t er;

	er = cudaFree(sps.energy);
	er = cudaFree(sps.position);
	er = cudaFree(sps.velocity);
	er = cudaFree(sps.remain_time);
	er = cudaFree(sps.rand_data);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Memory Free Error: %s\n", cudaGetErrorString(err));
}

void copy_to_device_sps(simpleParticleSystem &sps) {
	cudaError_t err = cudaMemcpyToSymbol(d_sps, &sps, sizeof(simpleParticleSystem));

	if (err != cudaSuccess)
		printf("Constant Memory Copy Error: %s\n", cudaGetErrorString(err));
}

void generate_particles(int thread_size) {
	generateParticles << < 1, thread_size >> > ();
	//generateParticlesLine <<< 1, sps.ONE_BATCH_PARTICLE_SIZE >>> (
	//	sps.position, sps.velocity_orientation, sps.velocity, sps.remain_time, sps.rand_data, sps.ONE_BATCH_PARTICLE_SIZE,
	//	sps.MAX_PARTICLE_SIZE, sps.generator_line[0], sps.generator_line[1], sps.MAX_VELOCITY, sps.MIN_VELOCITY, sps.LIFE_TIME
	//);
}

void render_particles(uchar4* devPtr, int img_width, int img_height) {
	int thread_dim = 16;
	int grid_dim_x = (img_width + thread_dim - 1) / thread_dim;
	int grid_dim_y = (img_height + thread_dim - 1) / thread_dim;
	dim3 grids(grid_dim_x, grid_dim_y);
	dim3 threads(thread_dim, thread_dim);
	renderParticles << <grids, threads >> > (devPtr, img_width, img_height);
}

__global__ void generateParticles()
{
	float2 *position = (*d_sps).position;
	float2 *velocity = (*d_sps).velocity;
	float *remain_time = (*d_sps).remain_time;
	float *rand = (*d_sps).rand_data;
	int generate_size = (*d_sps).ONE_BATCH_PARTICLE_SIZE;
	int max_size = (*d_sps).MAX_PARTICLE_SIZE;

	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ unsigned int generate_start_index;

	//get the particle generate block pos
	if (index == 0) {
		max_size -= generate_size;
		generate_start_index = 0;
		while (generate_start_index <= max_size) {
			if (remain_time[generate_start_index] == 0)
				break;
			generate_start_index += generate_size;
		}
	}

	__syncthreads();

	if (generate_start_index > max_size)
		return;

	int pid = generate_start_index + index; 
	float x;
	float y;
	float2 velocity_orientation;
	float n_velocity;

	//generate rand position and velocity
	switch ((*d_sps).TYPE)
	{
	case LineGenerator:
		x = rand[index] * ((*d_sps).generator_line[0].x - (*d_sps).generator_line[1].x) + (*d_sps).generator_line[1].x;
		y = rand[index] * ((*d_sps).generator_line[0].y - (*d_sps).generator_line[1].y) + (*d_sps).generator_line[1].y;
		position[pid] = make_float2(x, y);

		rand += generate_size;
		pid = generate_start_index + index;
		velocity_orientation = get_normal_vector(rand[index]);

		rand += generate_size;
		n_velocity = rand[index] * ((*d_sps).MAX_VELOCITY - (*d_sps).MIN_VELOCITY) + (*d_sps).MIN_VELOCITY;
		velocity[pid].x = n_velocity * velocity_orientation.x;
		velocity[pid].y = n_velocity * velocity_orientation.y;
		break;
	case CircleGenerator:
		float rand_pos = rand[index];
		float2 vec = get_normal_vector(rand_pos);
		x = vec.x * (*d_sps).generator_radius.x + (*d_sps).generator_center.x;
		y = vec.y * (*d_sps).generator_radius.y + (*d_sps).generator_center.y;
		position[pid] = make_float2(x, y);

		rand += generate_size;
		pid = generate_start_index + index;
		float rand_orient = rand[index];
		rand_orient = rand_pos + (rand_orient / 2 - rand_orient / 4);
		velocity_orientation = get_normal_vector(rand_orient);

		rand += generate_size;
		n_velocity = rand[index] * ((*d_sps).MAX_VELOCITY - (*d_sps).MIN_VELOCITY) + (*d_sps).MIN_VELOCITY;
		velocity[pid].x = n_velocity * velocity_orientation.x;
		velocity[pid].y = n_velocity * velocity_orientation.y;
		break;
	default:
		break;
	}

	//generate remain time
	remain_time[pid] = (*d_sps).LIFE_TIME;
}

__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= img_width || y >= img_height)
		return;

	
	if (!(x >= (*d_sps).BOUND_BOX[0] && x <= (*d_sps).BOUND_BOX[2]
		&& y <= (*d_sps).BOUND_BOX[1] && y >= (*d_sps).BOUND_BOX[3]))
		return;

	int generate_size = (*d_sps).ONE_BATCH_PARTICLE_SIZE;
	int max_size = (*d_sps).MAX_PARTICLE_SIZE;
	float energy = 0;
	float dist_bound_powerd = (*d_sps).ENERGY_SCOPE * (*d_sps).ENERGY_SCOPE;
	float2 pos = make_float2(x, y);
	for (int start_index = 0; start_index < max_size - generate_size; start_index += generate_size)
	{
		if ((*d_sps).remain_time[start_index] == 0)
			continue;
		//here we do not render the first particle of the batch
		for (int index = start_index + 1; index < start_index + generate_size; ++index) {
			if ((*d_sps).remain_time[index] != 0) {
				energy += get_energy((*d_sps).position[index], pos, dist_bound_powerd);
				if (energy >= 1) {
					energy = 1;
					break;
				}
			}
		}
		if (energy >= 1) {
			break;
		}
	}


	int offset = x + y * img_width;
	devPtr[offset] = get_color_from_energy(energy);
}


__global__ void updateParticles(float passed_time) {
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int strip = gridDim.x * blockDim.x;
	unsigned int start_index = blockIdx.x*blockDim.x;

	__shared__ int living_particle_num;

	while (index < (*d_sps).MAX_PARTICLE_SIZE) {
		living_particle_num = 0;
		__syncthreads();

		if ((*d_sps).remain_time[start_index] == 0)
			continue;
		
		float2 acc = get_acceleration(index);

		index += strip;
		start_index += strip;
	}
}


__device__ float2 get_normal_vector(float rand_num) {
	float x, y;
	sincosf(rand_num*2*PI, &y, &x);

	return make_float2(x, y);
}

__device__ float get_energy(float2 p1, float2 p2, float dist_bound_powerd) {
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;
	float dist_powered = dx*dx + dy*dy;

	if (dist_powered > dist_bound_powerd)
		return 0;
	if (dist_powered == 0)
		return 0.5;
	return 0.5 / dist_powered;
}

__device__ uchar4 get_color_from_energy(float energy) {
	unsigned char r = 255 * energy;
	unsigned char g = 180 * energy;
	unsigned char b = 60 * energy;
	unsigned char w = 255 * energy;

	return make_uchar4(r, g, b, w);
}

__device__ float2 get_acceleration(int index) {
	return make_float2(20.0, 0);
}

__device__ void update_particle_velocity(int index, float2 acc, float passed_time) {
	(*d_sps).velocity[index].x += acc.x * passed_time;
	(*d_sps).velocity[index].y += acc.y * passed_time;
}

__device__ int update_particle_position(int index, float passed_time) {
	if ((*d_sps).remain_time[index] - passed_time <= 0) {
		(*d_sps).remain_time[index] = 0;
		return 0;
	}
	
	float2 *pos = &(*d_sps).position[index];
	(*pos).x += (*d_sps).velocity[index].x * passed_time;
	(*pos).y += (*d_sps).velocity[index].y * passed_time;

	if ((*pos).x > (*d_sps).LIFE_BOUND[0] && (*pos).x < (*d_sps).LIFE_BOUND[2]
		&& (*pos).y < (*d_sps).LIFE_BOUND[1] && (*pos).y < (*d_sps).LIFE_BOUND[3]) {
		return 1;
	}

	(*d_sps).remain_time[index] = 0;
	return 0;
}