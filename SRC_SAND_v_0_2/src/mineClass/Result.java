package mineClass;

public class Result implements Comparable{
    
    double Prediction = 0;
    double TrueClass = 0;
    boolean IsOutlier = false;    
    public int Id = -1; //instance number
    double Dist = 0; //distance
    
    /** Creates a new instance of Result */
    public Result() {
    }
    
    public Result(int id, double pred, double truecls, boolean outlier)
    {
        Id = id;
        Prediction = pred;
        TrueClass = truecls;
        IsOutlier = outlier;
    }
    
    public int compareTo(Object o)
    {
        if(Dist > ((Result)o).Dist)
            return 1;
        else if (Dist < ((Result)o).Dist)
            return -1;
        return 0;
    }        
}
