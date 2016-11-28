package reasc;

import mineClass.Constants;

public class Datapoint{
    public double[] avector = null; // attribute vector
    int clusterId = Constants.NIL;  // Id of current cluster 
    int classId = Constants.NIL;    // class label 
    int trueId = Constants.NIL;     // true class
    boolean labeled = false; //is it a labeled data?
    long Id = Constants.NIL; //Unique Id of this datapoint
        
    public Datapoint(double[] av, String label, int ctrId, boolean labld, long id)
    {
        //avector = new double[av.length];
        avector = av.clone();
        trueId = Constants.getLabelId(label);
        clusterId = ctrId;        
        labeled = labld;
        Id = id;        
    }
    
    public Datapoint(double[] av, int label, int ctrId, boolean labld, long id)
    {
        //avector = new double[av.length];
        avector = av.clone();
        trueId = label;
        clusterId = ctrId;        
        labeled = labld;
        Id = id;        
    }
    
    public Datapoint(Datapoint p)
    {
        avector = new double[p.avector.length];
        avector = p.avector.clone();
        trueId = p.trueId;
        clusterId = p.clusterId;        
        classId = p.classId;
        labeled = p.labeled;
        Id = p.Id;        
    }
    
    public static double Euclidean(Datapoint p1, Datapoint p2)
    {
        int i = 0;
        double d = 0;
        for(i = 0; i < p1.avector.length; i ++)
            d = d + (p1.avector[i] - p2.avector[i]) * (p1.avector[i] - p2.avector[i]);
        return Math.sqrt(d);
    }
}

