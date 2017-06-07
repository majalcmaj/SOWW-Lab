// #include <pthread.h>
#include<omp.h>
#include <stdio.h> 
#include <stdlib.h>
#include <mpi.h> 
#define THREADNUM 4
#define RESULT 1
pthread_t thread[THREADNUM];
pthread_attr_t attr;
int startValue[THREADNUM];
double precision=1000000000; 
//2000000000; 
int step;
double pilocal=0;
int myrank,proccount; 
void *Calculate(int myrank, int thread_id) {
	int start = myrank * THREADNUM + thread_id; // start from this number
	int mine,sign; 
	double pi=0;
	int count=0;
	// each process performs computations on its part                                                                                
	pi=0; 
	mine=start*2+1; 
	sign=(((mine-1)/2)%2)?-1:1; 
	for (;mine<precision;) { 
		/*
		   if (!(count%1000000)) {
		   printf("\nProcess %d %ld %ld", myrank,sign,mine);                                                                       
		   fflush(stdout);         
		   }                                                                                                                                                          
		 */
		pi+=sign/(double)mine; 
		mine+=2*step; 
		sign=(((mine-1)/2)%2)?-1:1; 
		count++;
	} 
	MPI_Send(&pi,1,MPI_DOUBLE,0,RESULT,MPI_COMM_WORLD);
	/*
	// now update our local process pi value
	pilocal+=pi;
	 */
}
int main(int argc, char **argv) { 
	double pi_final=0; 
	int i,j; 
	int threadsupport;
	void *threadstatus;
	MPI_Status status;
	// Initialize MPI                                                                                                                                     
	MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&threadsupport); 
	if (threadsupport!=MPI_THREAD_MULTIPLE) {
		printf("\nThe implementation does not support MPI_THREAD_MULTIPLE, it supports level %d\n",threadsupport);
		MPI_Finalize();
		exit(-1);
	}
	// find out my rank                                                                                                                                
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	// find out the number of processes in MPI_COMM_WORLD                                                           
	MPI_Comm_size(MPI_COMM_WORLD, &proccount); 
	// now distribute the required precision                                                                                                
	if (precision<proccount) { 
		printf("Precision smaller than the number of processes - try again."); 
		MPI_Finalize(); 
		return -1; 
	} 
	// now start the threads in each process
	// define the thread as joinable
	// pthread_attr_init(&attr);
	// pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	// initialize the step value
	step=proccount*THREADNUM;
#pragma omp parallel firstprivate(i, myrank) num_threads(THREADNUM)
	{
		i = omp_get_thread_num();
		Calculate(myrank, i); 
	}
	if (!myrank) { // receive results from the threads
		double resulttemp;
		for(i=0;i<proccount;i++)
			for(j=0;j<THREADNUM;j++)  {
				MPI_Recv(&resulttemp,1,MPI_DOUBLE,i,RESULT,MPI_COMM_WORLD,&status);
				printf("\nReceived result %f from thread %d process %d",resulttemp,j,i);
				fflush(stdout);
				pi_final+=resulttemp;
			}
	}
	/*
	// now merge the numbers to rank 0                                                                                                     
	MPI_Reduce(&pilocal,&pi_final,1, 
	MPI_DOUBLE,MPI_SUM,0, 
	MPI_COMM_WORLD); 
	 */
	if (!myrank) { 
		pi_final*=4; 
		printf("pi=%f",pi_final);
	}	


	// Shut down MPI                                                                                                                                  
	MPI_Finalize(); 
	return 0; 
} 
