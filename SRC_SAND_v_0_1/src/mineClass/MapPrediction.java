/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

//saves the predictions of each classifier to an instances
package mineClass;

public class MapPrediction{
    public int Cid = -1;               //classifier id
    public double Predclass = -1;      //the prediction
    public boolean Isoutlier = false;  //Outlier?
    public double Dist = 0;            //distance of the outlier
    public int NumIsNovel = 0;         //how many (models) say it is novel class?
    /* fields for weighted voting */
    public double distFromPer = 0.0;
    public double ratioMaxClass = 0.0; //ratio of instances supporting max class in the closest cluster
    public double weight = -1.0; //weight = -1.0 means weight is not actually calculated for this prediction
    /* fields for weighted voting */

    public MapPrediction(int cid, double predclass, boolean outlier, double dist)
    {
        Cid = cid;
        Predclass = predclass;
        Isoutlier = outlier;
        Dist = dist;
    }
    
    public MapPrediction(int cid, double predclass, boolean outlier, double dist, double distFromPer, /*double ratioOfLabeledInst, */double ratioMaxClass, double weight)
    {
        this.Cid = cid;
        this.Predclass = predclass;
        this.Isoutlier = outlier;
        this.Dist = dist;
        this.distFromPer = distFromPer;
        this.ratioMaxClass = ratioMaxClass;
        this.weight = weight;
    }
}
