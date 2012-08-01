#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

// Number of threads
#define NUM_THREADS 32

//OpenMP chunk size
#define CHUNK_SIZE 10

// Number of iterations
#define TIMES 1

// Input Size
#define NSIZE 7
#define NMAX 262144

int Ns[NSIZE] = {4096, 8192, 16384, 32768, 65536, 131072, 262144};  
int length; 

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

//test file
FILE *input, *output;

/* Copies source data to target arrays */
void init(int n){
	int i, chunk, num_threads;
	
	chunk = CHUNK_SIZE;
	num_threads = NUM_THREADS;

	#pragma omp parallel for \
	 shared (A, B, n, chunk, num_threads) private (i) \
	 schedule (static, chunk) \
	 num_threads (num_threads)

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

/* Sequential recursive algorithm to solve prefix/suffix minima problem for given array 
 * Enhanced with OpenMP pragmas to share the worlkload
 * source - source array,
 * length - array length,
 * prefix = 1 for prefix minima, 0 for suffix minima
 */
void scan_seq(int* source, int length, int prefix){
	int i, chunk, num_threads;
	int Z[NMAX];

	if (1 == length){
		return;
	}

	chunk = CHUNK_SIZE;
	num_threads = NUM_THREADS;

	printf("Source: (%d) ", length);
	print_array(source, length);

	//computing sums of consequitive pairs of items
	#pragma omp parallel for \
	 shared (Z, source, chunk, num_threads) private (i) \
	 schedule (static, chunk) \
	 num_threads (num_threads)

	for (i = 0; i < length; i++){
		if (0 == i % 2){
			Z[i/2] = min(source[i], source[i+1]);
		}
	}

	//recursively computes prefix/suffix sums of obtained sequence
	scan_seq(Z, length/2, prefix);

	printf("Z done: ");
	print_array(Z, length/2);

	//filling in the odd values from the obtained array
	if (1 == prefix){
		//prefix minima
		#pragma omp parallel for \
	 	 shared (Z, source, chunk, num_threads) private (i) \
	 	 schedule (static, chunk) \
	 	 num_threads (num_threads)

		for (i = 1; i < length; i++){
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

		for (i = length - 2; i >= 0; i--){
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
 * length - array length,
 * prefix = 1 for prefix minima, 0 for suffix minima
 */
void scan_par(int* source, int length, int prefix){	
	int i;
	int Z[NMAX];

	if (1 == length){
		return;
	}

	printf("Source: (%d) ", length);
	print_array(source, length);

	//computing sums of consequitive pairs of items
	for (i = 0; i < length; i++){
		if (0 == i % 2){
			Z[i/2] = min(source[i], source[i+1]);
		}
	}

	//recursively computes prefix/suffix sums of obtained sequence
	scan_par(Z, length/2, prefix);

	printf("Z done: ");
	print_array(Z, length/2);

	//filling in the odd values from the obtained array
	if (1 == prefix){
		//prefix minima
		for (i = 1; i < length; i++){
			if (1 == i % 2){
				source[i] = Z[i/2];
			} else {
				source[i] = min(source[i],Z[i/2 - 1]);
			}
		}
	} else if (0 == prefix){
		//suffix minima
		for (i = length - 2; i >= 0; i--){
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

/* Sequential algotithm implementation */
void seq_function(int n){
	printf("\n");
	printf("Sequential algorithm\n");

	scan_seq(B, n, 1);

	printf("Prefix: ");
	print_array(B, n);

	init(n);

	scan_seq(B, n, 0);
	printf("Suffix: ");
	print_array(B, n);
	
}

/* Returns minimum of two given integers */
int min(int a, int b){
	if (a < b){
		return a;
	}
	return b;
}

void* par_function(void* a){

	int *n;

	n = (int *)a;

	printf("\n");
	printf("Parallel algorithm\n");

	scan_par(B, *n, 1);

	printf("Prefix: ");
	print_array(B, *n);

	init(*n);

	scan_par(B, *n, 0);
	printf("Suffix: ");
	print_array(B, *n);

}

/* Reads a source array from a file into array A */
void read_file(char* file)
{
	int current_num;
	int i;

	input = fopen(file, "r");

	if (NULL == input)
	{
		printf("Cannot open test file\n");
		exit(1);
	}

	i = 0;
	while(fscanf(input, "%d ", &current_num) != EOF){
		A[i++] = current_num;
	}
	length = i;

	fclose(input);
}

/* Runs the program */
int main (int argc, char *argv[])
{
	struct timeval startt, endt, result;
	int i, j, k, nt, t, n, c;
	void *status;
	pthread_attr_t attr;
	tThreadArg x[NUM_THREADS];
	
	result.tv_sec = 0;
	result.tv_usec= 0;

	/* Generate a seed input */
	// srand ( time(NULL) );
	// for(k=0; k<NMAX; k++){
	// 	A[k] = rand();
	// }

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
		printf("| %d | %d |",n,TIMES);

		/* Run sequential algorithm */
		result.tv_usec=0;
		gettimeofday (&startt, NULL);
		for (t = 0; t < TIMES; t++) {
			init(n);
			seq_function(length);
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
			printf("test\n");
			result.tv_sec=0; result.tv_usec=0;
			for (j=1; j<=/*NUMTHRDS*/nt; j++)
			{
				x[j].id = j; 
				x[j].nrT=nt; // number of threads in this round
				x[j].n=n;  //input size
				pthread_create(&callThd[j-1], &attr, par_function, (void *)&x[j]);
			}

			gettimeofday (&startt, NULL);
			for (t=0; t<TIMES; t++) 
			{
				init(n);
				pthread_barrier_wait(&barr);
			}
			gettimeofday (&endt, NULL);

			/* Wait on the other threads */
			for(j=0; j</*NUMTHRDS*/nt; j++)
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
