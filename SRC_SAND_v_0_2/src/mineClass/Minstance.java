/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mineClass;

import weka.core.Instance;
import java.util.ArrayList;

public class Minstance extends Instance{
    public int Id = -1;                             //Unique ID
    public ArrayList Predictions = new ArrayList(); //Indivisual predictions of the classifiers
    public ArrayList OPredictions = new ArrayList();//Indivisual predictions of the olindda models
    public MapPrediction EPrediction = null;        //Ensemble prediction
    public int ClusterId = -1;                      //Which cluster (Fpseudopoint) does it belong to
    public boolean Comitted = false;                //Classification done?
    public boolean err = false, fp = false, fn = false, tp = false;
    public boolean isNovel = false;                 //is it a novel class instance during comit?  

    public Minstance(Instance inst, int id)
    {
        super(inst);
        this.m_Dataset = inst.dataset();
        this.Id = id;
    }

    //add a prediction of a classifier
    public void addPrediction(int cid, double predclass, boolean outlier, double dist)
    {        
        Predictions.add(new MapPrediction(cid, predclass, outlier, dist));
    }
    
    public void addPrediction(int cid, double predclass, boolean outlier, double dist, double distFromPer, /*double ratioOfLabeledInst, */double ratioMaxClass, double weight)
    {        
        Predictions.add(new MapPrediction(cid, predclass, outlier, dist , distFromPer, /*ratioOfLabeledInst, */ratioMaxClass, weight));
    }

    public void addPrediction(MapPrediction mp)
    {
        Predictions.add(mp);
    }
    
    public void clearPredictions()
    {
        Predictions.clear();
        OPredictions.clear();
        EPrediction = null;
        ClusterId = -1;                      //Which cluster (Fpseudopoint) does it belong to
    }

    //get the prediction of a classifier
    public MapPrediction getPrediction(int cid)
    {
        for(int i = 0; i < Predictions.size(); i ++)
        {
            MapPrediction mp = (MapPrediction)Predictions.get(i);
            if(mp.Cid == cid)
                return mp;
        }
        return null; //error
    }
}


