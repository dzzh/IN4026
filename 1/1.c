#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>

// Number of threads
#define NUM_THREADS 32

//OpenMP chunk size
#define CHUNK_SIZE 10

// Number of iterations
#define TIMES 1

// Input Size
#define NSIZE 7
#define NMAX 262144

#define STACK_SIZE_MB 32

// Prints results to standart output
// #define OUTPUT

//Prints debug data
//#define DEBUG

int Ns[NSIZE] = {4096, 8192, 16384, 32768, 65536, 131072, 262144};  

typedef struct __ThreadArg {
	int id;
	int nrT;
	int n;
} tThreadArg;

pthread_t callThd[NUM_THREADS];
pthread_mutex_t mutexpm;
pthread_barrier_t barr, internal_barr;

// Seed Input
int A[NMAX];

// Prefix/suffix subset
int B[NMAX];

//data file
FILE *input;

//input length
int length; 

/*
 * Utility functions
 */

/* Copies source data to target array for processing */
void init(int n){
	int i;

	//int chunk, num_threads;
	
	// chunk = CHUNK_SIZE;
	// num_threads = NUM_THREADS;

	// #pragma omp parallel for \
	//  shared (A, B, n, chunk, num_threads) private (i) \
	//  schedule (static, chunk) \
	//  num_threads (num_threads)

	for (i = 0; i < n; i++){
		B[i] = A[i];
	}
}

/* Prints given array in one row */
void print_array(int* array, int length){
	int i;

	for(i = 0; i < length; i++){
		printf("%d ", array[i]);
	}

	printf("\n");
}

/* Returns minimum of two given integers */
int min(int a, int b){
	if (a < b){
		return a;
	}
	return b;
}

/* Reads data from a submitted file into array A */
void read_file(char* file)
{
	int current_num;
	int i;

	input = fopen(file, "r");

	if (NULL == input)
	{
		printf("Cannot open data file\n");
		exit(1);
	}

	i = 0;
	while(fscanf(input, "%d ", &current_num) != EOF){
		A[i++] = current_num;
	}

	length = i;

	printf("Length: %d\n",length);
	fclose(input);
}

/* Increases stack size to the given value in MBytes */
void set_stack_size(int mb){
	rlim_t kStackSize = mb * 1024 * 1024;
	struct rlimit rl;
    int result;
	
	/* Increases stack size*/
    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
            }
        }
    }
}

/* Sequential recursive algorithm to solve prefix/suffix minima problem for given array 
 * enhanced with OpenMP pragmas to share the worlkload
 * source - source array,
 * len - array length,
 * prefix = 1 for prefix minima, 0 for suffix minima
 */
void scan_seq(int* source, int len, int prefix){
	int i, chunk, num_threads;
	int Z[NMAX];

	//exit condition
	if (1 == len){
		return;
	}

	chunk = CHUNK_SIZE;
	num_threads = NUM_THREADS;

	#ifdef DEBUG
		printf("Source: (%d) ", len);
		print_array(source, len);
	#endif

	//computing sums of consequitive pairs of items
	#pragma omp parallel for \
	 shared (Z, source, chunk, num_threads) private (i) \
	 schedule (static, chunk) \
	 num_threads (num_threads)

	for (i = 0; i < len; i++){
		if (0 == i % 2){
			Z[i/2] = min(source[i], source[i+1]);
		}
	}

	//recursively computes prefix/suffix sums of obtained sequence Z
	scan_seq(Z, len/2, prefix);

	#ifdef DEBUG
		printf("Z done: ");
		print_array(Z, len/2);
	#endif

	//filling in the odd values from the obtained array
	if (1 == prefix){
		//prefix minima
		#pragma omp parallel for \
	 	 shared (Z, source, chunk, num_threads) private (i) \
	 	 schedule (static, chunk) \
	 	 num_threads (num_threads)

		for (i = 1; i < len; i++){
			if (1 == i % 2){
				source[i] = Z[i/2];
			} else {
				source[i] = min(source[i],Z[i/2 - 1]);
			}
		}

	} else if (0 == prefix){
		//suffix minima
		#pragma omp parallel for \
	 	 shared (Z, source, chunk, num_threads) private (i) \
	 	 schedule (static, chunk) \
	 	 num_threads (num_threads)

		for (i = len - 2; i >= 0; i--){
			if (0 == i % 2){
				source[i] = Z[i/2];
			} else {
				source[i] = min(source[i],Z[i/2 + 1]);
			}
		}

	} else {
		//unrecognised prefix value
		exit(1);
	}
}

/* Parallel recursive algorithm to solve prefix/suffix minima problem for given array 
 * Pthreads are used for parallelization
 * source - source array,
 * len - array length,
 * prefix = 1 for prefix minima, 0 for suffix minima
 */
void scan_par(int* source, int len, int prefix){	
	int i;
	int Z[NMAX];

	//exit condition
	if (1 == len){
		return;
	}

	#ifdef DEBUG
		printf("Source: (%d) ", len);
		print_array(source, len);
	#endif

	//computing sums of consequitive pairs of items
	for (i = 0; i < len; i++){
		if (0 == i % 2){
			Z[i/2] = min(source[i], source[i+1]);
		}
	}

	//recursively computes prefix/suffix sums of obtained sequence Z
	scan_par(Z, len/2, prefix);

	#ifdef DEBUG
		printf("Z done: ");
		print_array(Z, len/2);
	#endif

	//filling in the odd values from the obtained array
	if (1 == prefix){
		//prefix minima
		for (i = 1; i < len; i++){
			if (1 == i % 2){
				source[i] = Z[i/2];
			} else {
				source[i] = min(source[i],Z[i/2 - 1]);
			}
		}

	} else if (0 == prefix){
		//suffix minima
		for (i = len - 2; i >= 0; i--){
			if (0 == i % 2){
				source[i] = Z[i/2];
			} else {
				source[i] = min(source[i],Z[i/2 + 1]);
			}
		}

	} else {
		//unrecognised prefix value
		exit(1);
	}
}

/* Sequential algorithm invocation */
void seq_function(int n){

	//prefix minima
	scan_seq(B, n, 1);

	#ifdef OUTPUT
		printf("Prefix minima: ");
		print_array(B, n);
	#endif

	//B reinitialization
	init(n);

	//suffix minima
	scan_seq(B, n, 0);

	#ifdef OUTPUT
		printf("Suffix minima: ");
		print_array(B, n);
	#endif
	
}

/* Parallel algorithm invocation */
void *par_function(void* a){

	pthread_barrier_wait(&barr);
	//pthread_barrier_wait(&internal_barr);

	tThreadArg *thread_data;

	thread_data = (tThreadArg *)a;

	// printf("Id: %d\n", thread_data->id);
	// printf("NrT: %d\n", thread_data->nrT);
	// printf("N: %d\n", thread_data->n);
	//prefix minima
	//scan_par(B, *n, 1);

	// #ifdef OUTPUT
	// 	printf("Prefix: ");
	// 	// print_array(B, thread_data.n);
	// #endif	

	//reinitialization of B
	//init(*n);

	//suffix minima 
	//scan_par(B, *n, 0);

	// #ifdef OUTPUT
	// 	printf("Suffix: ");
	// 	// print_array(B, thread_data.n);
	// #endif	
	//printf("Test2\n");
	return NULL;	
}

/* Runs the program */
int main (int argc, char *argv[])
{
	struct timeval startt, endt, result;
	int i, j, k, nt, t, n, c;
	void *status;
	pthread_attr_t attr;
	tThreadArg x[NUM_THREADS];

	set_stack_size(STACK_SIZE_MB);

	result.tv_sec = 0;
	result.tv_usec= 0;

	if (argc < 2){
		printf("Please specify the data file\n");
	}

	/* Read seed input from the file */
	read_file(argv[1]);

   	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	printf("|NSize|Iterations|Seq|Th01|Th02|Th04|Th08|Par16|\n");

	// for each input size
	for(c = 0; c < NSIZE; c++){
		n=Ns[c];
		printf("| %d | %d |", n, TIMES);

		/* Run sequential algorithm */
		result.tv_usec=0;
		gettimeofday (&startt, NULL);

		for (t = 0; t < TIMES; t++) {
			init(n);
			seq_function(n);
		}

		gettimeofday (&endt, NULL);
		result.tv_usec = (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
		printf(" %ld.%06ld | ", result.tv_usec/1000000, result.tv_usec%1000000);

		/* Run threaded algorithm(s) */
		for(nt = 1; nt < NUM_THREADS; nt = nt << 1){
			if(pthread_barrier_init(&barr, NULL, nt+1))
			{
				printf("Could not create a barrier\n");
				return -1;
			}
			if(pthread_barrier_init(&internal_barr, NULL, nt))
			{
				printf("Could not create a barrier\n");
				return -1;
			}
			result.tv_sec=0; result.tv_usec=0;
			for (j = 1; j <= nt; j++)
			{
				x[j].id = j; 
				x[j].nrT=nt; // number of threads in this round
				x[j].n=n;  //input size
				pthread_create(&callThd[j-1], &attr, par_function, (void *)&x[j]);
			}

			gettimeofday (&startt, NULL);
			for (t = 0; t < TIMES; t++) 
			{
				init(n);
				pthread_barrier_wait(&barr);
			}
			gettimeofday (&endt, NULL);

			/* Wait on the other threads */
			for(j = 0; j < nt; j++)
			{
				pthread_join(callThd[j], &status);
			}

			if (pthread_barrier_destroy(&barr)) {
				printf("Could not destroy the barrier\n");
				return -1;
			}
			if (pthread_barrier_destroy(&internal_barr)) {
				printf("Could not destroy the barrier\n");
				return -1;
			}
			result.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
			printf(" %ld.%06ld | ", result.tv_usec/1000000, result.tv_usec%1000000);
		}
		printf("\n");
	}
	pthread_exit(NULL);
}
