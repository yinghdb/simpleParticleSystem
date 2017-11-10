#define FREEGLUT_STATIC
#define GLEW_STATIC

#include "simple_particle.cuh"
#include <curand.h>
#include <stdio.h>      
#include <stdlib.h>     
#include "GL\glew.h"
#include "GL\glut.h"
#include <time.h>
#include <cuda_gl_interop.h>

#define DIM_X 720
#define DIM_Y 720
#define MIN(x,y) (x>y?y:x)
#define MAX(x,y) (x>y?x:y)

GLuint bufferObj;
cudaGraphicsResource *resource;
int term = 0;
uchar4* devPtr;
clock_t CPU_time;

void init_particle_system_line(simpleParticleSystem &sps)
{
	sps.TYPE = LineGenerator;
	sps.generator_line[0] = make_float2(DIM_X / 2, DIM_Y / 4);
	sps.generator_line[1] = make_float2(DIM_X / 2, DIM_Y / 2);

	sps.MAX_PARTICLE_SIZE = 51200;
	sps.ONE_BATCH_PARTICLE_SIZE = 256;
	sps.ENERGY_SCOPE = 5.0;
	sps.LIFE_BOUND[0] = 0;
	sps.LIFE_BOUND[1] = DIM_Y;
	sps.LIFE_BOUND[2] = DIM_X;
	sps.LIFE_BOUND[3] = 0;
	sps.BOUND_BOX[0] = MIN(sps.generator_line[0].x, sps.generator_line[1].x) - sps.ENERGY_SCOPE;
	sps.BOUND_BOX[1] = MAX(sps.generator_line[0].y, sps.generator_line[1].y) + sps.ENERGY_SCOPE;
	sps.BOUND_BOX[2] = MAX(sps.generator_line[0].x, sps.generator_line[1].x) + sps.ENERGY_SCOPE;
	sps.BOUND_BOX[3] = MIN(sps.generator_line[0].y, sps.generator_line[1].y) - sps.ENERGY_SCOPE;
	sps.MAX_VELOCITY = 100.0;
	sps.MIN_VELOCITY = 10.0;
	sps.LIFE_TIME = 3.0;
}

void init_particle_system_circle(simpleParticleSystem &sps)
{
	sps.TYPE = CircleGenerator;
	sps.generator_center = make_float2(DIM_X / 2, DIM_Y / 2);
	sps.generator_radius = make_float2(40, 40);

	sps.MAX_PARTICLE_SIZE = 51200;
	sps.ONE_BATCH_PARTICLE_SIZE = 256;
	sps.ENERGY_SCOPE = 3.0;
	sps.LIFE_BOUND[0] = 0;
	sps.LIFE_BOUND[1] = DIM_Y;
	sps.LIFE_BOUND[2] = DIM_X;
	sps.LIFE_BOUND[3] = 0;
	sps.BOUND_BOX[0] = sps.generator_center.x - sps.generator_radius.x - sps.ENERGY_SCOPE;
	sps.BOUND_BOX[1] = sps.generator_center.y + sps.generator_radius.y + sps.ENERGY_SCOPE;
	sps.BOUND_BOX[2] = sps.generator_center.x + sps.generator_radius.x + sps.ENERGY_SCOPE;
	sps.BOUND_BOX[3] = sps.generator_center.y - sps.generator_radius.y - sps.ENERGY_SCOPE;
	sps.MAX_VELOCITY = 100.0;
	sps.MIN_VELOCITY = 10.0;
	sps.LIFE_TIME = 3.0;
}

void randomGenerator(float* devData, int number, unsigned long long seed) {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandGenerateUniform(gen, devData, number);
	curandDestroyGenerator(gen);
}

//openGL render functions
void drawFunc(void)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(DIM_X, DIM_Y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

static void keyFunc(unsigned char key, int x, int y)
{
	switch (key) {
	case 27:
		cudaGraphicsUnregisterResource(resource);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

int init_cuda_gl(int argc, char* argv[]) {
	// ����һ���豸���Զ���prop    
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));

	//�޶��豸���㹦�ܼ��İ汾��    
	prop.major = 1;
	prop.minor = 0;

	//ѡ���ڼ��㹦�ܼ��İ汾��Ϊ1.0��GPU�豸������    
	cudaChooseDevice(&dev, &prop);

	//ѡ��GL�������е��豸    
	cudaGLSetGLDevice(dev);

	//OpenGL������ʼ��    
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM_X, DIM_Y);
	glutCreateWindow("CUDA+OpenGL");

	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		return -1;
	}

	// �������ػ���������
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM_X*DIM_Y * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	// imgId����ʱ����CUDA��OpenGL�乲��ͨ����imgIdע��Ϊһ��ͼ����Դ
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}

void particle_update() {
	// ӳ��ù�����Դ 
	cudaGraphicsMapResources(1, &resource, NULL);
	// ����һ��ָ��ӳ����Դ��ָ��
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	render_particles(devPtr, DIM_X, DIM_Y);

	// ȡ��ӳ�䣬ȷ��cudaGraphicsUnmapResource()֮ǰ������CUDA�������
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

int main(int argc, char* argv[]) {
	simpleParticleSystem sps;

	//init meta data
	init_particle_system_circle(sps);
	//gpu memory malloc
	init_particles_cuda(sps);
	//bound d_sps
	copy_to_device_sps(sps);

	//init first batch particles
	randomGenerator(sps.rand_data, 3 * sps.ONE_BATCH_PARTICLE_SIZE, 12345LL);
	generate_particles(sps.ONE_BATCH_PARTICLE_SIZE);

	//init opengl
	init_cuda_gl(argc, argv);

	//update particles
	particle_update();

	glutKeyboardFunc(keyFunc);
	glutDisplayFunc(drawFunc);
	//glutIdleFunc(idleFunc);
	glutMainLoop();

	//destory gpu memory
	destroy_particles_cuda(sps);


	return 0;
}