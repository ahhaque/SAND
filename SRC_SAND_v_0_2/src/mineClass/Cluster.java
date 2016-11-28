package mineClass;

import weka.core.Instance;

public class Cluster {
    
    
    public Instance centroid = null;    
    public int n = 0;               //total number of points in the cluster
    public double radius = 0;       //maximum distance from a data point to the centroid
    public double meand = 0;               //mean distance from the centroid
    double vard = 0;                //variance of distance from the centroid
    public double sumx2 = 0, sumx = 0;     //for computing variance
    public double[] dist = null;    //distribution of different classes
    public int majority = -1;       //majority class
    public double sc = -2;          //silhouette coefficient    
            
    //instantiate a new cluster on a given centroid
    public Cluster(Instance c, int cltrId)
    {
        centroid = new Instance(c);     
        centroid.setDataset(c.dataset());
        //clusterId = cltrId;
        dist = new double[c.numClasses()];        
        //sum = new Instance(c);
        
    }   
    
    public Cluster(Cluster c)
    {
        centroid = new Instance(c.centroid);
        centroid.setDataset(c.centroid.dataset());        
        //clusterId = c.clusterId;
        if(c.dist != null)
            dist = c.dist.clone();
    
        n = c.n;
        radius = c.radius;
        meand = c.meand;
        vard = c.vard;
        sumx2 = c.sumx2;
        majority = c.majority;    
    }
        
    void init()
    {            
        n = 0;      radius = 0;
        meand = 0;  vard = 0;
        sumx2 = 0;  sumx = 0;
        majority = -1;
        
        for(int i = 0; i < dist.length; i ++)
            dist[i] = 0;   
    }            
}//Cluster
