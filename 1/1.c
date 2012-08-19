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

//Recommendation of least number of operations
//performed by a thread to decrease thread creation overhead
//for small tasks
#define THREAD_OPS 128

//OpenMP chunk size
#define CHUNK_SIZE 10

// Number of iterations
#define TIMES 3

// Input Size
#define NSIZE 7
#define NMAX 262144

#define STACK_SIZE_MB 2

//Prints results to standart output
//#define OUTPUT

//Prints debug data
//#define DEBUG

//Performs sequential computations
#define SEQUENTIAL

//Use OpenMP optimizations to improve 
//the performance of the sequential solution
#define OPENMP

//Performs parallel computations
#define PARALLEL

int Ns[NSIZE] = {4096, 8192, 16384, 32768, 65536, 131072, 262144};
//int Ns[NSIZE] = {32, 64, 128, 256, 512, 1024, 2048};
//int Ns[NSIZE] = {32};

typedef struct __ThreadArg {
	int id;
	int nrT;
	int n;
	int prefix;
	int* a;
	int* b;
} tThreadArg;

pthread_t callThd[NUM_THREADS];
pthread_mutex_t mutexpm;
pthread_attr_t attr;
pthread_barrier_t barr, internal_barr;

// Seed Input
int A[NMAX];

// Prefix/suffix subset
int B[NMAX];

//data file
FILE *input;

//input length
int length;

//recommended minimum number of operations for the thread
int thread_ops;

/*
 * Utility functions
 */

/* Copies source data to target array for processing */
void init(int n) {
	int i;

	for (i = 0; i < n; i++) {
		B[i] = A[i];
	}
}

/* Prints given array in one row */
void print_array(int* array, int length) {
	int i;

	for(i = 0; i < length; i++) {
		printf("%d ", array[i]);
	}

	printf("\n");
}

/* Returns minimum of two given integers */
int min(int a, int b) {
	if (a < b) {
		return a;
	}

	return b;
}

/* Reads data from a submitted file into array A */
void read_file(char* file) {
	int current_num;
	int i;
	input = fopen(file, "r");

	if (NULL == input) {
		printf("Cannot open data file\n");
		exit(1);
	}

	i = 0;

	while(fscanf(input, "%d ", &current_num) != EOF) {
		A[i++] = current_num;
	}

	length = i;
	printf("Length: %d\n", length);
	fclose(input);
}

/* Increases stack size to the given value in MBytes */
void set_stack_size(int mb) {
	rlim_t kStackSize = mb * 1024 * 1024;
	struct rlimit rl;
	int result;
	/* Increases stack size*/
	result = getrlimit(RLIMIT_STACK, &rl);

	if (result == 0) {
		if (rl.rlim_cur < kStackSize) {
			rl.rlim_cur = kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);

			if (result != 0) {
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

//Calculates optimal number of threads to perform the computation
//based on max number of available threads, n input size and
//recommended minimum of operations per one thread
int get_optimal_threads_number(int max, int n) {
	int opt;

	opt = max;

	while (opt * thread_ops > n && opt > 1) {
		opt /= 2;
	}

	return opt;
}

/* Sequential recursive algorithm to solve prefix/suffix minima problem for
 * given array enhanced with OpenMP pragmas to share the worlkload.
 * source - source array,
 * len - array length,
 * prefix = 1 for prefix minima, 0 for suffix minima
 */
void scan_seq(int* source, int len, int prefix) {
	int i, chunk, num_threads;
	int Z[len];

	//exit condition
	if (1 == len) {
		return;
	}

	chunk = CHUNK_SIZE;
	num_threads = NUM_THREADS;

	#ifdef DEBUG
		printf("Source: (%d) ", len);
		print_array(source, len);
	#endif

	//computing sums of consequitive pairs of items
	#ifdef OPENMP	
		#pragma omp parallel for \
		shared (Z, source, chunk, num_threads) private (i) \
		schedule (static, chunk) \
		num_threads (num_threads)
	#endif

	for (i = 0; i < len; i++) {
		if (0 == i % 2) {
			Z[i / 2] = min(source[i], source[i + 1]);
		}
	}

	//recursively computes prefix/suffix sums of obtained sequence Z
	scan_seq(Z, len / 2, prefix);

	#ifdef DEBUG
		printf("Z done: ");
		print_array(Z, len / 2);
	#endif

	//filling in the odd values from the obtained array
	if (1 == prefix) {
		//prefix minima
		#ifdef OPENMP
			#pragma omp parallel for \
			shared (Z, source, chunk, num_threads) private (i) \
			schedule (static, chunk) \
			num_threads (num_threads)
		#endif

		for (i = 1; i < len; i++) {
			if (1 == i % 2) {
				source[i] = Z[i / 2];
			} else {
				source[i] = min(source[i], Z[i / 2 - 1]);
			}
		}

	} else if (0 == prefix) {
		//suffix minima
		#ifdef OPENMP
			#pragma omp parallel for \
			shared (Z, source, chunk, num_threads) private (i) \
			schedule (static, chunk) \
			num_threads (num_threads)
		#endif

		for (i = len - 2; i >= 0; i--) {
			if (0 == i % 2) {
				source[i] = Z[i / 2];
			} else {
				source[i] = min(source[i], Z[i / 2 + 1]);
			}
		}

	} else {
		//unrecognised prefix value
		exit(1);
	}
}

/* Threaded function to compute sums of consequitive pairs of items
 * in a part of given array */
void* par_sum(void* a) {
	tThreadArg *thread_data;
	int i, first_index, last_index, size;

	thread_data = (tThreadArg *)a;

	//compute size of the subspace to work with 
	//and indices of the first and last elements in the subspace
	size = (int)(thread_data->n / thread_data->nrT);
	first_index = (thread_data->id - 1) * size;
	last_index = first_index + size; //non-inclusive

	for (i = first_index; i < last_index; i++) {
		if (0 == i % 2) {
			thread_data->b[i/2] = min(thread_data->a[i], thread_data->a[i+1]);
		}
	}

	return NULL;
}

/* Threaded function to fill in odd values 
 * in prefix/suffix sums computations */
void* par_odd(void* a) {
	tThreadArg *thread_data;
	int i, size, first_index, last_index, prefix;
	int* source;
	int* Z;

	thread_data = (tThreadArg *)a;

	//compute size of the subspace to work with and index
	size = (int)(thread_data->n / thread_data->nrT);
	first_index = (thread_data->id - 1) * size;
	last_index = first_index + size; //non-inclusive
	prefix = thread_data->prefix;
	source = thread_data->a;
	Z = thread_data->b;

	//filling in the odd values from the obtained array
	if (1 == prefix) {
		//prefix minima
		for (i = first_index; i < last_index; i++) {

			if (1 == i % 2) {
				// printf("i = %d, source[i] = %d, z[i/2] = %d\n",i,source[i],Z[i/2]);
				source[i] = Z[i / 2];
			} else if (0 != i) {
				// printf("else i = %d, source[i] = %d, z[i/2-1] = %d\n",i,source[i],Z[i/2-1]);
				source[i] = min(source[i], Z[i / 2 - 1]);
			}
		}

	} else if (0 == prefix) {
		//suffix minima
		for (i = last_index-1; i >= first_index; i--) {

			if (i > thread_data->n - 2){
				continue;
			}

			if (0 == i % 2) {
				source[i] = Z[i / 2];
			} else {
				source[i] = min(source[i], Z[i / 2 + 1]);
			}
		}

	} else {
		//unrecognised prefix value
		exit(1);
	}

	return NULL;
}

/* Parallel recursive algorithm to solve prefix/suffix minima 
 * problem for given array, pthreads are used for parallelization.
 * source - source array,
 * len - array length,
 * nt - number of threads,
 * prefix = 1 for prefix minima, 0 for suffix minima */
void scan_par(int* source, int len, int nt, int prefix) {
	int i, j;
	int Z[len];
	int opt;
	void *status;
	tThreadArg x[NUM_THREADS];

	//exit condition
	if (1 == len) {
		return;
	}

	#ifdef DEBUG
		printf("Source: (%d) ", len);
		print_array(source, len);
	#endif

	opt = get_optimal_threads_number(nt, len);

	//multi-threaded sum computation of consequitive pairs of items
	for (j = 1; j <= opt; j++) {
		x[j].id = j;
		x[j].nrT = opt; // number of threads in this round
		x[j].n = len; //input size
		x[j].prefix = prefix;
		x[j].a = source;
		x[j].b = Z;
		pthread_create(&callThd[j - 1], &attr, par_sum, (void *)&x[j]);
	}

	/* Wait on completion */
	for(j = 0; j < opt; j++) {
		pthread_join(callThd[j], &status);
	}

	//recursively computes prefix/suffix sums of obtained sequence Z
	scan_par(Z, len / 2, nt, prefix);

#ifdef DEBUG
	printf("Z done: ");
	print_array(Z, len / 2);
#endif

	//filling in odd values
	for (j = 1; j <= opt; j++) {
		pthread_create(&callThd[j - 1], &attr, par_odd, (void *)&x[j]);
	}

	/* Wait on completion */
	for(j = 0; j < opt; j++) {
		pthread_join(callThd[j], &status);
	}

}

/* Sequential algorithm invocation
   n - number of elements */
void seq_function(int n) {
	//prefix minima
	scan_seq(B, n, 1);

#ifdef OUTPUT
	printf("\nPrefix minima: ");
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

/* Parallel algorithm invocation
   n - number of elements
   nt - number of threads */
void par_function(int n, int nt) {

	//prefix minima
	scan_par(B, n, nt, 1);

#ifdef OUTPUT
	printf("\nPrefix: ");
	print_array(B, n);
#endif

	//reinitialization of B
	init(n);

	//suffix minima
	scan_par(B, n, nt, 0);

#ifdef OUTPUT
	printf("Suffix: ");
	print_array(B, n);
#endif
}

/* Runs the program */
int main (int argc, char *argv[]) {
	struct timeval startt, endt, result;
	int k, nt, t, n, c;
	set_stack_size(STACK_SIZE_MB);
	result.tv_sec = 0;
	result.tv_usec = 0;

	if (argc < 2) {
		printf("Please specify the data file\n");
	}

	/* Read seed input from the file */
	read_file(argv[1]);

	//setting recommended minimum number of thread operations
	if (3 == argc){
		thread_ops == (int)argv[2];
	} else {
		thread_ops = THREAD_OPS;
	}

	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	printf("|NSize|Iterations|Seq|Th01|Th02|Th04|Th08|Par16|\n");

	// for each input size
	for(c = 0; c < NSIZE; c++) {
		n = Ns[c];
		printf("| %d | %d |", n, TIMES);
		/* Run sequential algorithm */
		result.tv_usec = 0;
		gettimeofday (&startt, NULL);

#ifdef SEQUENTIAL

		for (t = 0; t < TIMES; t++) {
			init(n);
			seq_function(n);
		}

#endif

		gettimeofday (&endt, NULL);
		result.tv_usec = (endt.tv_sec * 1000000 + endt.tv_usec) - (startt.tv_sec * 1000000 + startt.tv_usec);
		printf(" %ld.%06ld | ", result.tv_usec / 1000000, result.tv_usec % 1000000);

		/* Run threaded algorithm(s) */

		// if(pthread_barrier_init(&barr, NULL, nt + 1)) {
		// 	printf("Could not create a barrier\n");
		// 	return -1;
		// }

		// if(pthread_barrier_init(&internal_barr, NULL, nt)) {
		// 	printf("Could not create a barrier\n");
		// 	return -1;
		// }

		for(nt = 1; nt < NUM_THREADS; nt = nt << 1) {
			result.tv_sec = 0;
			result.tv_usec = 0;
			gettimeofday (&startt, NULL);

#ifdef PARALLEL

			for (t = 0; t < TIMES; t++) {
				init(n);
				par_function(n, nt);
			}

#endif

			gettimeofday (&endt, NULL);
			result.tv_usec += (endt.tv_sec * 1000000 + endt.tv_usec) -
			                  (startt.tv_sec * 1000000 + startt.tv_usec);
			printf(" %ld.%06ld | ", result.tv_usec / 1000000,
			       result.tv_usec % 1000000);
		}

		// if (pthread_barrier_destroy(&barr)) {
		// 	printf("Could not destroy the barrier\n");
		// 	return -1;
		// }

		// if (pthread_barrier_destroy(&internal_barr)) {
		// 	printf("Could not destroy the barrier\n");
		// 	return -1;
		// }

		printf("\n");
	}

	pthread_exit(NULL);
}
