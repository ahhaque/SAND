/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mineClass;

import java.util.ArrayList;

/**
 *
 * @author axh129430
 */
public class ConfidenceStats {
    public ArrayList<Double> corrCoeff;
    public ArrayList<Double> minConfVals;
    public ArrayList<Double> maxConfVals;
    public double accuracy; //accuracy of this classifier on last training chunk.
    
    public ConfidenceStats()
    {
        corrCoeff = new ArrayList<Double>();
        minConfVals = new ArrayList<Double>();
        maxConfVals = new ArrayList<Double>();
        accuracy = 0.0;
    }
    
    public ArrayList<Double> getCorrCoeff()
    {
        return this.corrCoeff;
    }
    
    public ArrayList<Double> getMinConfVals()
    {
        return this.minConfVals;
    }
    
    public ArrayList<Double> getMaxConfVals()
    {
        return this.maxConfVals;
    }   
    
    public double getAccuracy()
    {
        return this.accuracy;
    }
    
    public void clearContents()
    {
        this.corrCoeff.clear();
        this.minConfVals.clear();
        this.maxConfVals.clear();
        this.accuracy = 0.0;
    }
}
