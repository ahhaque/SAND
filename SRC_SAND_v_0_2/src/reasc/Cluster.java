package reasc;

import mineClass.Constants;

public class Cluster {
    
    /** Creates a new instance of Cluster */
   
    //static int C = 0;
    public Datapoint centroid = null;
    public int clusterId = Constants.NIL;                
    public int majority = Constants.NIL; //majority class of this cluster
    public double[] frq = null;    //frequency of labeled points of different classes in this cluster    
    public double[] sum = null;    //sum of all data vectors
    //ArrayList[] ldata = null; //labeled data points    
    //ArrayList data = new ArrayList(); //all unlabeled data points     
    public int n = 0;              //total number of points in the cluster
    public double lp = 0;          //total number labeled of points in the cluster
    public double radius = 0;      //radius of the cluster
   
    //instantiate a new cluster on a given centroid
    public Cluster(Datapoint c, int cltrId)
    {
        centroid = new Datapoint(c.avector, "", 0, false, Constants.NIL);                
        clusterId = cltrId;        
        //ldata = new ArrayList[ReascCore.C];
        frq = new double[ReascCore.C];
        sum = new double[c.avector.length];
    }   
    
    //copy a cluster (need only to copy the summary statistics)
    public Cluster(Cluster c)
    {
        centroid = new Datapoint(c.centroid.avector, "", 0, false, Constants.NIL);                
        clusterId = c.clusterId;                
        
        frq = c.frq.clone();
        sum = c.sum.clone();    
                        
        majority = c.majority;
        
        n = c.n;
        lp = c.lp;
        radius = c.radius;
    }
    
    void init()
    {            
        for(int i = 0; i < frq.length; i ++)
        {         
            frq[i] = 0;
        }    
        
        for(int i = 0; i < sum.length; i ++)
            sum[i] = 0;        
    }
    
    public void addpoint(Datapoint p)
    {
        if(p.labeled == true)
        {            
            frq[p.classId]++;
            lp ++;
        }
        n ++;
        
        //update summary
        for(int i = 0; i < sum.length; i ++)
            sum[i] += p.avector[i];
        
    }//add point    
       
    public void removepoint(Datapoint p)
    {
        if(p.labeled == true)
        {
            //ldata[p.classId].remove(getIndex(p,ldata[p.classId]));            
            frq[p.classId] --;
            lp --;
        }
        n --;
        
        //update summary
        for(int i = 0; i < sum.length; i ++)
            sum[i] -= p.avector[i];
    }//remove point
    
    void compfreq() //compute frequency of labled points
    {        
        double max = 0;
        for(int i = 0; i < frq.length; i ++)
        {
            //frq[i] = ldata[i].size();
            if(max < frq[i])
            {
                max = frq[i];
                majority = i;
            }
        }
        
    }//compfreq
    
  
}
