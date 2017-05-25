#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>

#define PACKET_COUNT 10000
#define RESOLUTION 1000


typedef struct {
	void *data;
} t_data;


typedef struct {
	void *result;
} t_result;


double dtime()
{
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,(struct timezone*)0);
	tseconds = (double)(mytime.tv_sec +mytime.tv_usec*1.0e-6);
	return( tseconds );
}


t_data *partition(t_data inputdata,int partition_count) {}


double f(double x) {
	return sin(x)*sin(x)/x;
}


double one_shot_integration(double start,double end) {
	return ((end-start)*(f(start)+f(end))/2);
}


double adaptive_integration(double start,double end,int k) {
	// first check if we need to go any deeper
	double middle=(start+end)/2;
	double a1,a2,a;
	double r;
	int threadid;
	a=one_shot_integration(start,end);
	a1=one_shot_integration(start,middle);
	a2=one_shot_integration(middle,end);
	if (k<3) {
		if (fabs(a-a1-a2)>0.001) { // go deeper
			r=0;
#pragma omp parallel sections firstprivate(start,middle,end,k) reduction(+:r) num_threads(2)
			{
#pragma omp section
				{
					r=adaptive_integration(start,middle,k+1);
				}
#pragma omp section
				{
					r=adaptive_integration(middle,end,k+1);
				}
			}
			return r;
		}
	} else {
		if (fabs(a-a1-a2)>0.001) { // go deeper
			a1=adaptive_integration(start,middle,k+1);
			a2=adaptive_integration(middle,end,k+1);
			a=a1+a2;
		}
	}
	return a;
}


void process(t_data data,t_result result) {
	// use the adaptive quadrature approach i.e. check whether we need
	// to dive deeper or whether the current resolution if fine
	// this can also be parallelized using OpenMP - later put it into a framework
	// compute an integrate of function f()
	// over range [data.data[0], data.data[1]]
	// do it adaptively
	double pivot;
	double r_a=((double *)data.data)[0];
	double r_b=((double *)data.data)[1];
	double integrate=0;
	double r_length=(r_b-r_a)/RESOLUTION;
	double x=r_a;
	int i;
	double t_start,t_stop;
	t_start=dtime();
	integrate=adaptive_integration(r_a,r_b,0);
	*((double *)result.result)=integrate; // store the result
}


t_result *allocate_results(int *packet_count) {
	int i;
	// prepare space for results
	t_result *r_results=(t_result *)malloc(sizeof(t_result)*PACKET_COUNT);
	if (r_results==NULL) {
		perror("Not enough memory");
		exit(-1);
	}
	for(i=0;i<PACKET_COUNT;i++) {
		r_results[i].result=malloc(sizeof(double));
		if (r_results[i].result==NULL) {
			perror("Not enough memory");
			exit(-1);
		}
	}
	*packet_count=PACKET_COUNT;
	return r_results;
}


t_data *generate_data(const int packet_count, double packet_size, double a) {
	// prepare the input data
	// i.e. the given range is to be divided into packets
	int i;

	t_data *p_packets;
	double *p_data;
	double r_a,r_b;
	// prepare PACKET_COUNT number of packets
	t_data *packets=(t_data *)malloc(sizeof(t_data)*packet_count);
	if (packets==NULL) {
		perror("Not enough memory");
		exit(-1);
	}
	r_a=a;
	r_b=a+packet_size;
	p_packets=packets; // pointer to the beginning of the packets
	for(i=0;i<packet_count;i++) {
		packets[i].data=malloc(2*sizeof(double));
		if (packets[i].data==NULL) {
			perror("Not enough memory");
			exit(-1);
		}
		// populate the packet with the data
		p_data=(double *)packets[i].data;
		*p_data=r_a;
		*(p_data+1)=r_b;
		r_a+=packet_size;
		r_b+=packet_size;
	}
	// *packet_count=PACKET_COUNT;
	return packets;
}


t_data *data;
t_result *results;

main(int argc,char **argv) {
	// prepare the input data
	// i.e. the given range is to be divided into packets
	// now the main processing loop using OpenMP
	int counter;
	int my_data;
	double result;
	int i;
	int iter;
	int tmp;
	int packet_count = PACKET_COUNT;
	int results_count;
	int threadid;
	int myrank, commsize;
	int threadclass; // for using various critical sections
	double t_start,t_stop,t_total,t_current=0,t_min;
	int t_counter=0;
	t_min=100000000;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &tmp);

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);

	double result_tmp;
	double a=1,b=100000000;
	double packet_size=(b-a)/PACKET_COUNT;
	const int number_of_iterations = 20;
	int dest;
	result=0;
	if(myrank == 0) 
	{
#pragma omp parallel num_threads(commsize - 1) private (a, dest) reduction(+:result)
		{
			dest = omp_get_thread_num() + 1;
#pragma omp for
			for(iter = 0 ; iter < number_of_iterations; iter++) {
				packet_count = PACKET_COUNT / number_of_iterations;
				a = 1.0 + ((double)(iter*packet_count)*packet_size);

				MPI_Send(&packet_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
				MPI_Send(&a, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
				MPI_Recv(&result_tmp, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				result += result_tmp;
			}	
			printf("Sending finishing signl to %d\n\n", dest);
			packet_count = 0;
			//for(iter= 1 ; iter < commsize; iter++)
			MPI_Send(&packet_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		}
		printf("\nFinished");
		printf("\nThe total value of the integrate is %.5f\n",result);
	}
	else {
		while(1) {
			MPI_Recv(&packet_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(packet_count == 0) {
				printf("Slave %d finished.", myrank);
				break;
			}
			MPI_Recv(&a, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			data=generate_data(packet_count, packet_size, a);
			results=allocate_results(&results_count);
			counter=0;
			t_total=0;
			t_start=dtime();
			omp_set_nested(1);
			omp_set_dynamic(0);
#pragma omp parallel private(my_data) firstprivate(threadid,threadclass) shared(counter) num_threads(60)
			{
				threadid=omp_get_thread_num();
				do {
					// each thread will try to get its data from the available list
#pragma omp critical
					{
						my_data=counter;
						counter++; // also write result to the counter
					}
					// process and store result -- this can be done without synchronization
					//results[my_data]=
					if (my_data<packet_count)
						process(data[my_data],results[my_data]); // note that processing
					// may take various times for various data packets
				} while (my_data<packet_count); // otherwise simply exit because
				// there are no more data packets to process
			}
			t_stop=dtime();
#pragma omp barrier
			result = 0;
			// now just add results
			for(i=0;i<packet_count;i++) {
				// printf("\nVal[%d]=%.8f",i,*((double *)results[i].result));
				// fflush(stdout);
				result+=*((double *)results[i].result);
			}
			t_counter++;
			t_current=t_stop-t_start;
			if (t_current<t_min)
				t_min=t_current;

			MPI_Send( &result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
		}
	}

	MPI_Finalize();
}



