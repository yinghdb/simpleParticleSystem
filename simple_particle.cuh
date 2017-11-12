#include "cuda_runtime.h"

#define PARTICLE_LIFE_MS 3000
#define PI 3.1415926

enum ParticleGeneratorType
{
	LineGenerator,
	CircleGenerator
};

typedef struct
{
	////physical attribute////
	float2 *position;
	float2 *velocity;
	float *energy;

	////particle life////
	float *remain_time;

	////rand data used for generating particles////
	float *rand_data;

	////common attributes////
	int MAX_PARTICLE_SIZE;
	int ONE_BATCH_PARTICLE_SIZE;
	float ENERGY_SCOPE;
	int LIFE_BOUND[4]; //left, top, right, bottom
	int BOUND_BOX[4];
	float MAX_VELOCITY; //pixels per second
	float MIN_VELOCITY;
	float LIFE_TIME; //second

	/////particle generator type related////
	enum ParticleGeneratorType TYPE;
	//LineGenerator
	float2 generator_line[2];
	//CircleGenerator
	float2 generator_center;
	float2 generator_radius;

} simpleParticleSystem;

void init_particles_cuda(simpleParticleSystem &sps);
void destroy_particles_cuda(simpleParticleSystem &sps);
void generate_particles(int generate_size);
void copy_to_device_sps(simpleParticleSystem &sps);
void render_particles(uchar4* devPtr, int img_width, int img_height);
void updata_particles(int generate_size, float passed_time);