package mineClass;

import java.util.Arrays;
import weka.core.Instance;
import weka.core.Instances;

public class Outlier {
    
    double[] PrInterval = {50,75,87.5,93.75,96.875,98.4375,99.21875}; 
    //probability profile intervals (7 intervals) denoting 50% of data in that interval
    public double[] PrValue = new double[7]; 
    //values for each interval
    
    public static final int OUTLIER_CLASS = -100; //class label for the outlier class
    /** Creates a new instance of Outlier */
    public Outlier() {
    }
    
    public double Prob(Instance p, Cluster c)
    {
        return Kmeans.Euclidean(p,c.centroid,-1);
        //return Math.exp(-1.0 * Kmeans.Euclidean(p,c.centroid,-1));
    }
    
  
    
    public double[] checkOutlier(Instance instance, Cluster[] clusters)
    {
        //find the nearest cluster from instance
        
        double min = Kmeans.Euclidean(instance,clusters[0].centroid,-1);
        int minj = 0;
        for(int i = 1; i < clusters.length; i ++)
        {
            double d = Kmeans.Euclidean(instance,clusters[i].centroid,min);
            if( d < min)
            {
                min = d;
                minj = i;
            }
        }
        double[] ret = new double[2];
        //EvoDT.logger.debug("distance= "+min+"  nearest Cluster= "+minj+ " ");
        if(min > clusters[minj].radius ) //distance greater than radius
        {
            ret[0] = 1; //true
            ret[1] = min - clusters[minj].radius;
            return ret;
        }
        else
        {
            ret[0] = 0;
            ret[1] = 0;
            return ret;
        }
    }        
    
      
    //compute neighborhood silhouette coefficient with each outlier clusters considered as
    //separate cluster (not pseudopoint)
    //m = nearest neighbors outlierId = cluster Ids of the outlier data points
    public boolean getNSCseparate(Cluster[] clusters, Cluster[] outliers, int m) 
    {
                
        //First, compute S.C. corresponding to each pseudopoint
        Result[][] result = new Result[outliers.length][];        
        double[] a = new double[outliers.length];
        int tot = 0; //total points whose S.C. > 0
        
        //get pair-wise distances
        for(int i = 0; i < outliers.length; i ++)
        {
            result[i] = new Result[outliers.length];
            for(int j = 0; j < outliers.length; j ++)
            {
                result[i][j] = new Result(i,0,0,false);
                result[i][j].Dist = Kmeans.Euclidean(outliers[i].centroid, outliers[j].centroid,-1);
            }
        }
        
        //sort
        for(int i = 0; i < outliers.length; i ++)
        {
            Arrays.sort(result[i]);
        }
        
        //find the average distance of each outlier points to other m outlier points
        
        for(int i = 0; i < outliers.length; i ++)
        {
            //find the nearest m data  points            
            double wt = 0;
            a[i] = 0;
            int j = 0;
            while(wt < m && j < outliers.length)
            {
                int id = result[i][j].Id; //get the next nearest outlier cluster
                wt += outliers[id].n; 
                
                if(id == i) //own cluster
                {
                    a[i] += outliers[id].sumx;  //sum of distance from each point to the centroid                            
                }
                else
                {
                    a[i] += 
                     Kmeans.Euclidean(outliers[i].centroid, outliers[id].centroid, -1) * outliers[id].n ;
                }
                j ++;
                
            }//while
            
            //compute average
            if(wt > 0)
                a[i] /= wt;
        }//for
        
        //find silhouette coefficient of each outlier point, get the avg s.c. of the pseudopoint
        //filter out negative pseudopoints        
        for(int i = 0; i < outliers.length; i ++)
        {            
            int C = clusters[0].centroid.numClasses();
            double[] b = new double[C]; //weighted sum of distances to each class
            double[] wt = new double[C];//sum of weight of each class
            Result[] clResult = new Result[clusters.length];
            
            //first sort them according to distances
            for(int j = 0; j < clusters.length; j ++)
            {
                clResult[j] = new Result(j,0,0,false);
                clResult[j].Dist = Kmeans.Euclidean(outliers[i].centroid, clusters[j].centroid, -1);
            }
            
            Arrays.sort(clResult);
            
            //process according to increasing order of distance
            for(int j = 0; j < clusters.length; j ++)
            {
                int id = clResult[j].Id;
                int c = clusters[id].majority;
                if(wt[c] < m)
                {
                    wt[c] += clusters[id].n;
                    b[c] += 
                      Kmeans.Euclidean(outliers[i].centroid, clusters[id].centroid, -1) * clusters[id].n;
                }//if
            }//for            
            
            for(int c = 0; c < C; c ++)
                if(wt[c] > 0)
                    b[c] /= wt[c];
            
            //get the minimum (not zero, not c)
            int c = C;
            int mj = -1;
            for(int j = 0; j < C; j ++)
            {
                if(wt[j] > 0 ) //only nonzero frq
                {
                    if(mj < 0) //not initialized yet
                    {
                        mj = j;
                    }
                    else
                    {
                        if(b[j] < b[mj] )
                            mj = j;
                    }//else
                }//if
            }//for j <= C
            
            double sc = (b[mj] - a[i])/Math.max(b[mj],a[i]);
            outliers[i].sc = sc;
            
            Constants.logger.debug("Outlier pseudopoint, silhouette coefficient=" + sc + ", Total points = "+outliers[i].n);
            
            if(sc > 0)
                tot += outliers[i].n;
        }//for i       
                        
        if( tot > m) 
            return true;
        else
            return false;
    }           
  
   
}
