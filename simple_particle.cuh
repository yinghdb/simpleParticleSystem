#include "cuda_runtime.h"

enum ParticleGeneratorType
{
	LineGenerator,
	CycleGenerator
};

typedef struct
{
	//physical attribute
	float3 *pose;
	char3 *color;
	float3 *velocity_orientation;
	float *velocity;
	float3 *acceleration_orientation;
	float *energy;

	//partile life
	int *remain_cycle;

	//common attributes
	float energy_scope_powered;
	float acceleration;
	enum ParticleGeneratorType type;

} simpleParticles;