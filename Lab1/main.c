#include <stdio.h> 
#include <mpi.h>
#include<unistd.h>
int main(int argc, char **argv) { 
	double precision=1000000000; 
	int myrank,proccount; 
	double pi,pi_final; 
	int mine,sign; 
	int i, iter; 
	// Initialize MPI
	MPI_Init(&argc, &argv); 
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &proccount); 
	printf("Rank: %d\nSize: %d\n", myrank, proccount);

	// now distribute the required precision
	if (precision<proccount) { 
		printf("Precision smaller than the number of processes - try again."); 
		MPI_Finalize(); 
		return -1; 
	} 
	// each process performs computations on its part
	pi=0; 
	iter = 0;
	mine=myrank*2+1; 
	sign=(((mine-1)/2)%2)?-1:1; 
	for (;mine<precision;) {
		fflush(stdout);
		pi+=sign/(double)mine; 
		mine+=2*proccount; 
		sign=(((mine-1)/2)%2)?-1:1; 
	}
	printf("Wchodze do sleep (%d)\n", myrank);
	if(myrank >= 3)
		sleep(5);
	printf("Opuszczam sleep (%d)\n", myrank);
	
	// now merge the numbers to rank 0
	MPI_Reduce(&pi,&pi_final,1, 
			MPI_DOUBLE,MPI_SUM,1, 
			MPI_COMM_WORLD); 
	if (myrank==1) { 
		pi_final*=4; 
		printf("pi=%f",pi_final); 
	} 
	// Shut down MPI
	MPI_Finalize(); 
	return 0; 
} 
