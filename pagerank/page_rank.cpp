#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{
 

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    int empty_outgoing=0;
    int* empty_outgoing_array =(int*) malloc(numNodes*sizeof(int));
    int start,end,node;


    for (int j = 0; j < numNodes; j++) {
      int start_out,end_out;
      solution[j] = equal_prob;
      start_out=g->outgoing_starts[j];
      end_out = (j == numNodes-1) ? g->num_edges : g->outgoing_starts[j+1];

      
      if((end_out-start_out)==0){
        empty_outgoing_array[empty_outgoing++]=j;
      }
    }
  
  
  
     /*TODO STUDENTS: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:*/

     // initialization: see example code above
     
    bool converged = false;


    double* score_new,*aux=NULL;
    int start_out,end_out;
    

    score_new =(double*) malloc(numNodes*sizeof(double));


    while (!converged) {

      double sum = 0;
      #pragma omp parallel for reduction(+:sum) schedule(dynamic,32)
      for (int j = 0; j < empty_outgoing; j++) {

          sum+=damping * solution[empty_outgoing_array[j]] / numNodes;
          
      } 
      
      #pragma omp parallel for schedule(dynamic,32)
      for (int i = 0; i < numNodes; i++ ){

        start=g->incoming_starts[i];
        end = (i == numNodes-1) ? g->num_edges : g->incoming_starts[i+1];

        score_new[i]=0;

        for (int j = start; j < end; j++) {

          node=g->incoming_edges[j];
          start_out=g->outgoing_starts[node];
          end_out = (node == numNodes-1) ? g->num_edges : g->outgoing_starts[node+1];
          

          score_new[i]+=solution[node]/(end_out-start_out);
         
        }

        score_new[i]=(damping * score_new[i]) + (1.0-damping) / numNodes;

        score_new[i]+=sum;

      }

      double global_diff=0;

      #pragma omp parallel for reduction(+:global_diff) schedule(dynamic,32)
      for (int i = 0; i < numNodes; i++) {

          // #pragma omp atomic update
          global_diff += fabs(score_new[i] - solution[i]) ;
        
      }

      // global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
      converged = (global_diff < convergence) ? true : false;


      #pragma omp parallel for schedule(dynamic,32)
      for (int i = 0; i < numNodes; i++) {
        solution[i] = score_new[i];
      }

    }

 
  free(score_new);
  free(empty_outgoing_array);

}
