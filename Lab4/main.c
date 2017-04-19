#include <stdio.h> 
#include <mpi.h> 
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#define PRECISION 0.000001
#define RANGESIZE 1
#define DATA 0
#define RESULT 1
#define FINISH 2
#define DEBUG
double f(double x) {
    return sin(x)*sin(x)/x;
}
double SimpleIntegration(double a,double b) {
    double i;
    double sum=0;
    for (i=a;i<b;i+=PRECISION)
        sum+=f(i)*PRECISION;
    return sum;
}
int main(int argc, char **argv) { 
    MPI_Request *requests;
    int requestcount=0;
    int requestcompleted;
    int myrank,proccount; 
    double a=1,b=100;
    double *ranges;
    double range[2];
    double result=0;
    double *resulttemp; 
    int sentcount=0;
    int recvcount=0;
    int i;
    int thread_available;
    MPI_Status status;
    // Initialize MPI                                                                                                                                     
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_available);
    if(!thread_available) {
        printf("Cannot start multithreading environment\n");
    } 
    // find out my rank                                                                                                                                
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
    // find out the number of processes in MPI_COMM_WORLD                                                           
    MPI_Comm_size(MPI_COMM_WORLD, &proccount); 
    if (proccount<2) {
        printf("Run with at least 2 processes");
        MPI_Finalize();
        return -1;
    }
    if (((b-a)/RANGESIZE)<2*(proccount-1)) {
        printf("More subranges needed");
        MPI_Finalize();
        return -1;
    }

# pragma omp parallel sections num_threads(2) default(none) \
    shared(myrank, a, b, proccount, stdout, ompi_mpi_comm_world, ompi_mpi_double, ompi_request_null) private(requests, i, resulttemp, sentcount, range, ranges, requestcompleted, result, recvcount, status)
    {
#pragma omp section
        { 
            printf("Section 1 myrank: %d omp_thread_num: %d\n", myrank, omp_get_thread_num());
            fflush(stdout);
            // now the master will distribute the data and slave processes will perform computations
            if (myrank==0) {
                requests=(MPI_Request *)malloc(3*proccount*sizeof(MPI_Request));
                if (!requests) {
                    printf("\nNot enough memory");
                    MPI_Finalize();
                }
                ranges=(double *)malloc(4*proccount*sizeof(double));
                if (!ranges) {
                    printf("\nNot enough memory");
                    MPI_Finalize();
                }
                resulttemp=(double *)malloc(proccount*sizeof(double));
                if (!resulttemp) {
                    printf("\nNot enough memory");
                    MPI_Finalize();
                }
                range[0]=a;
                // first distribute some ranges to all slaves
                for(i=0;i<proccount;i++) {
                    range[1]=range[0]+RANGESIZE;
#ifdef DEBUG
                    printf("\nMaster sending range %f,%f to process %d",range[0],range[1],i);
                    fflush(stdout);
#endif
                    // send it to process i
                    MPI_Send(range,2,MPI_DOUBLE,i,DATA,MPI_COMM_WORLD);
                    sentcount++;
                    range[0]=range[1];
                }
                // the first proccount requests will be for receiving, the latter ones for sending
                for(i=0;i<2*proccount;i++)
                    requests[i]=MPI_REQUEST_NULL; // none active at this point
                // start receiving for results from the slaves
                for(i=0;i<proccount;i++)
                    MPI_Irecv(&(resulttemp[i]),1,MPI_DOUBLE,i,RESULT,MPI_COMM_WORLD,&(requests[i]));
                // start sending new data parts to the slaves
                for(i=0;i<proccount;i++) {
                    range[1]=range[0]+RANGESIZE;
#ifdef DEBUG
                    printf("\nMaster sending range %f,%f to process %d",range[0],range[1],i);
                    fflush(stdout);
#endif
                    ranges[2*i]=range[0];
                    ranges[2*i+1]=range[1];
                    // send it to process i
                    MPI_Isend(&(ranges[2*i]),2,MPI_DOUBLE,i,DATA,MPI_COMM_WORLD,&(requests[proccount+i]));
                    sentcount++;
                    range[0]=range[1];
                }
                printf("Jestem1");
                while (range[1]<b) {
#ifdef DEBUG
                    printf("\nMaster waiting for completion of requests");
                    fflush(stdout);
#endif
                    // wait for completion of any of the requests
                    MPI_Waitany(2*proccount,requests,&requestcompleted,MPI_STATUS_IGNORE);
                    // if it is a result then send new data to the process
                    // and add the result
                    if (requestcompleted<proccount) {
                        result+=resulttemp[requestcompleted];
                        recvcount++;
#ifdef DEBUG
                        printf("\nMaster received %d result %f from process %d",recvcount,resulttemp[requestcompleted],requestcompleted);
                        fflush(stdout);
#endif
                        // first check if the send has terminated
                        MPI_Wait(&(requests[proccount+requestcompleted]),MPI_STATUS_IGNORE);
                        // now send some new data portion to this process
                        range[1]=range[0]+RANGESIZE;
                        if (range[1]>b) range[1]=b;
#ifdef DEBUG
                        printf("\nMaster sending range %f,%f to process %d",range[0],range[1],requestcompleted);
                        fflush(stdout);
#endif
                        ranges[2*requestcompleted]=range[0];
                        ranges[2*requestcompleted+1]=range[1];
                        MPI_Isend(&(ranges[2*requestcompleted]),2,MPI_DOUBLE,requestcompleted,DATA,MPI_COMM_WORLD,&(requests[proccount+requestcompleted]));
                        sentcount++;
                        range[0]=range[1];
                        // now issue a corresponding recv
                        MPI_Irecv(&(resulttemp[requestcompleted]),1,MPI_DOUBLE,requestcompleted,RESULT,MPI_COMM_WORLD,&(requests[requestcompleted]));
                    }
                }
                // now send the FINISHING ranges to the slaves
                // shut down the slaves
                range[0]=range[1];
                for(i=0;i<proccount;i++) {
#ifdef DEBUG
                    printf("\nMaster sending FINISHING range %f,%f to process %d",range[0],range[1],i);
                    fflush(stdout);
#endif
                    ranges[2*i+2*proccount]=range[0];
                    ranges[2*i+1+2*proccount]=range[1];
                    MPI_Isend(range,2,MPI_DOUBLE,i,DATA,MPI_COMM_WORLD,&(requests[2*proccount+1+i]));
                }
#ifdef DEBUG
                printf("\nMaster before MPI_Waitall with total proccount=%d",proccount);
                fflush(stdout);
#endif
                // now receive results from the processes - that is finalize the pending requests
                MPI_Waitall(3*proccount,requests,MPI_STATUSES_IGNORE);
#ifdef DEBUG
                printf("\nMaster after MPI_Waitall with total proccount=%d",proccount);
                fflush(stdout);
#endif
                // now simply add the results
                for(i=0;i<proccount;i++) {
                    result+=resulttemp[i];
                }
                // now receive results for the initial sends
                for(i=0;i<proccount;i++) {
#ifdef DEBUG
                    printf("\nMaster receiving result from process %d",i+1);
                    fflush(stdout);
#endif
                    MPI_Recv(&(resulttemp[i]),1,MPI_DOUBLE,i,RESULT,MPI_COMM_WORLD,&status);
                    result+=resulttemp[i];
                    recvcount++;
#ifdef DEBUG
                    printf("\nMaster received %d result %f from process %d",recvcount,resulttemp[i],i+1);
                    fflush(stdout);
#endif
                }
                // now display the result
                printf("\nHi, I am process 0, the result is %f\n",result); 
            }
        }
#pragma omp section
        {
            printf("Section 2 myrank: %d omp_thread_num: %d\n", myrank, omp_get_thread_num());
            fflush(stdout);

            requests=(MPI_Request *)malloc(2*sizeof(MPI_Request));
            if (!requests) {
                printf("\nNot enough memory");
                MPI_Finalize();
            }
            requests[0]=requests[1]=MPI_REQUEST_NULL;
            ranges=(double *)malloc(2*sizeof(double));
            if (!ranges) {
                printf("\nNot enough memory");
                MPI_Finalize();
            }
            resulttemp=(double *)malloc(2*sizeof(double));
            if (!resulttemp) {
                printf("\nNot enough memory");
                MPI_Finalize();
            }
            // first receive the initial data
            MPI_Recv(range,2,MPI_DOUBLE,0,DATA,MPI_COMM_WORLD,&status);
#ifndef DEBUG
            printf("\nSlave received range %f,%f",range[0],range[1]);
            fflush(stdout);
#endif
            while (range[0]<range[1]) { // if there is some data to process
                // before computing the next part start receiving a new data part
                MPI_Irecv(ranges,2,MPI_DOUBLE,0,DATA,MPI_COMM_WORLD,&(requests[0]));
                // compute my part
                resulttemp[1]=SimpleIntegration(range[0],range[1]);
#ifdef DEBUG
                printf("\nSlave just computed range %f,%f",range[0],range[1]);
                fflush(stdout);
#endif
                // now finish receiving the new part
                // and finish sending the previous results back to the master
                MPI_Waitall(2,requests,MPI_STATUSES_IGNORE);
#ifdef DEBUG
                printf("\nSlave just received range %f,%f",ranges[0],ranges[1]);
                fflush(stdout);
#endif
                range[0]=ranges[0];
                range[1]=ranges[1];
                resulttemp[0]=resulttemp[1];
                // and start sending the results back
                MPI_Isend(&resulttemp[0],1,MPI_DOUBLE,0,RESULT,MPI_COMM_WORLD,&(requests[1]));
#ifdef DEBUG
                printf("\nSlave just initiated send to master with result %f",resulttemp[0]);
                fflush(stdout);
#endif
            }
            // now finish sending the last results to the master
            MPI_Wait(&(requests[1]),MPI_STATUS_IGNORE);
        }
    }
    // Shut down MPI                                                                                                                                  
    MPI_Finalize(); 
#ifdef DEBUG
    printf("\nProcess %d finished",myrank);
    fflush(stdout);
#endif
    return 0; 
} 
