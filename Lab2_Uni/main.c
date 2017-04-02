#include <stdio.h> 
#include <mpi.h>

#define K 1
int main(int argc, char **argv) { 
	double precision=1000000000; 
	int myrank,proccount; 
	double pi,pi_final; 
	int mine = 0,sign; 
	int i, iteration, processed_element; 
	int N = 4 + ( K + 3 ) % 4;
	// Initialize MPI
	MPI_Init(&argc, &argv); 
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
	// each process performs computations on its part
	pi=0;

	
	iteration = 0;
	while( mine<precision ) {
		mine = 2 * (N * proccount * iteration + myrank * N) + 1;
		sign=(((mine-1)/2)%2)?-1:1; 
		for(i = 0; i < N && mine < precision; i++) {
			if (processed_element < 2 * N ) {
				printf("\nMyrank: %d iteracja: %d mine: %d sign: %d", myrank,processed_element, mine, sign);
				fflush(stdout);
			}
			pi+=sign/(double)mine; 
			mine+=2; 
			sign = -sign;
			processed_element++;
		}
		iteration ++;
	} 
	// now merge the numbers to rank 0
	MPI_Reduce(&pi,&pi_final,1, 
			MPI_DOUBLE,MPI_SUM,0, 
			MPI_COMM_WORLD); 
	if (!myrank) { 
		pi_final*=4; 
		printf("pi=%f",pi_final); 
	} 
	// Shut down MPI
	MPI_Finalize(); 
	return 0; 
} 
