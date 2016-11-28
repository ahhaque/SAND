package reasc;
     
public class Heap 
{    
      
      public Attrib[] A;           //heap array
      private int n = 0;            //num elements
      private int max = 0;         //Max elements
      
      Heap(int size)
      {
          A = new Attrib[size+1];          
          max = size;
      }
      //get number of elements
      public int heapSize()
      {
          return n;
      }
            
      void insert(double dist, double[] prob)
      {
          Attrib a = new Attrib(-dist, prob);
          
          //check whether the heapsize is maximum
          if(max > n)
          {
              //add the new element
              Insert(a);
          }
          else
          {
              //check with the minimum
              if(A[1].ig < a.ig) // then minimum must go
              {
                  ExtractMin();
                  Insert(a);
              }//if A
          }//else
      }//insert
      
      void Insert(Attrib a)
      {
          // insert at last
          A[++ n] = a;
          int r = n;
          if(r == 1) //root
              return;
          
          int u;
          while(r > 1)
          {
              u = r/2; //Parent;
              if(A[u].ig > A[r].ig) // parent is bigger. bubble up
              {
                  //swap elements
                  Attrib t = A[u]; A[u] = A[r]; A[r] = t;                  
                  r = u;
              }
              else break; 
          }//while r
      }//heapInsert
      
      void ExtractMin()
      {
          //first swap the first and last
          int r = 1, lst = n;
          
          //swap elements
          Attrib t = A[r]; A[r] = A[lst]; A[lst] = t;                  

          n --; //remove 
          
          while (r <= n / 2)            //bubble down
          {
              int s = r * 2;            //find the smaller of two children
              if( (s + 1 <= n) && (A[s + 1].ig < A[s].ig))
                  s = s + 1;
              if(A[r].ig > A[s].ig)     //greater than the smallest child
              {
                  //swap with smallest child               
                  t = A[r]; A[r] = A[s]; A[s] = t;                  
                  r = s;
              }
              else break;              
          }//while r <= n/2
      }//heapExtractMin
}
