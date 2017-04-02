#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define USE_MATH_DEFINES
#include <math.h>
#define PRECISION 0.00000001
#define RANGESIZE 0.0001
#define DATA 0
#define RESULT 1
#define FINISH 2
#define DEBUG
double f(double x) {
	return sin(x);
}
double SimpleIntegration(double a,double b) {
	double i;
	double sum=0;
	for (i=a;i<b;i+=PRECISION)
		sum+=f(i)*PRECISION;
	return sum;
}
int main(int argc, char **argv) { 
	int myrank,proccount; 
	double a=0,b=2 * M_PI;
	double range[2];
	double result=0,resulttemp; 
	int sentcount=0;
	int i; 
	MPI_Status status;
	// Initialize MPI
	MPI_Init(&argc, &argv); 
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &proccount); 
	if (proccount<2) {
		printf("Run with at least 2 processes");
		MPI_Finalize();
		return -1;
	}
	if (((b-a)/RANGESIZE) < 3*proccount) {
		printf("More subranges needed");
		MPI_Finalize();
		return -1;
	}
	// now the master will distribute the data and slave processes will perform computations
	if (myrank==0) {
		int finished_count;
		MPI_Request* requests = (MPI_Request*)malloc(sizeof(MPI_Request) * ( proccount - 1));
		int* finished_indices = (int*)malloc(sizeof(int) * ( proccount - 1 ));
		double* computation_results = (double*)malloc(sizeof(double) * (proccount - 1) ); 
		range[0]=a;
		// first distribute some ranges to all slaves
		for(i=1;i<proccount;i++) {
			range[1]=range[0]+ 2 * RANGESIZE;
#ifdef DEBUG
			printf("\nMaster sending range %f,%f to process %d",range[0],range[1],i);
			fflush(stdout);
#endif
			// send it to process i
			MPI_Send(range,2,MPI_DOUBLE,i,DATA,MPI_COMM_WORLD);
			sentcount++;
			range[0]=range[1];
			MPI_Irecv(computation_results + i - 1, 1, MPI_DOUBLE, i, RESULT, MPI_COMM_WORLD, requests + i - 1);
		}
		do {
			// Do root's computations
			range[1] = range[0] + RANGESIZE;
			if (range[1]>b) range[1]=b;
			result+=SimpleIntegration(range[0],range[1]);
			if (range[1]>b) break; // If root process took the last chunk of data
			// check the sender and send some more data
			range[0] = range[1];
			MPI_Testsome(proccount-1, requests, &finished_count,  finished_indices, MPI_STATUSES_IGNORE);
			if(finished_count > 0) {
				for(i = 0 ; i < finished_count ; i ++) {
					int index = finished_indices[i]; 
					result += computation_results[index];
					range[1]=range[0]+RANGESIZE;
					if (range[1]>b) range[1]=b;
					MPI_Send(range,2,MPI_DOUBLE,index + 1,DATA,MPI_COMM_WORLD);
					MPI_Irecv(computation_results + i, 1, MPI_DOUBLE, index + 1, RESULT, MPI_COMM_WORLD, requests + index);
					if (range[1]>b) break;
					range[0]=range[1];
				}
			}
		} while (range[1]<b);
		MPI_Waitall(proccount - 1, requests, MPI_STATUS_IGNORE);
		for(i = 0 ; i < proccount - 1; i ++) {
			result += computation_results[i];
		}

		// shut down the slaves
		for(i=1;i<proccount;i++) {
			MPI_Send(NULL,0,MPI_DOUBLE,i,FINISH,MPI_COMM_WORLD);
		}
		// now display the result
		printf("\nHi, I am process 0, the result is %f\n",result);	
		free(requests);
		free(finished_indices);
	       	free( computation_results );	
	} else { // slave
		// this is easy - just receive data and do the work
		do {
			MPI_Probe(0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			if (status.MPI_TAG==DATA) {
				MPI_Recv(range,2,MPI_DOUBLE,0,DATA,MPI_COMM_WORLD,&status);
				// compute my part
				resulttemp=SimpleIntegration(range[0],range[1]);
				// send the result back
				MPI_Send(&resulttemp,1,MPI_DOUBLE,0,RESULT,MPI_COMM_WORLD);
			}
		} while (status.MPI_TAG!=FINISH);
	}
	// Shut down MPI
	MPI_Finalize(); 
	return 0; 
} 
