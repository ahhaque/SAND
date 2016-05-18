/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package change_point_detection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import org.apache.commons.math.distribution.BetaDistributionImpl;

/**
 *
 * @author axh129430
 */
public class BetaDistributionChangePoint {
    private ArrayList<Double> dynamicWindow;
    private double gamma;
    private double sensitivity;
    int cushion;
    
    public BetaDistributionChangePoint(double gamma, double sensitivity)
    {
        dynamicWindow = new ArrayList<Double>();
        this.gamma = gamma;
        this.sensitivity = sensitivity;
    }
    
    public void insertIntoWindow(double value)
    {
        dynamicWindow.add(value);
    }
    
    public ArrayList<Double> getDynamicWindow()
    {
        return this.dynamicWindow;
    }
    
    public void shrinkWindow(int position /* inclusive */)
    {
        for(int index = 0; index <= position; index++)
        {
            dynamicWindow.remove(0);
        }
    }
    
    public void shrinkList(ArrayList<Double> list, int position /* inclusive */)
    {
        for(int index = 0; index <= position; index++)
        {
            list.remove(0);
        }
    }
    
    public int/*estimated change point*/ detectChange() throws Exception
    {
    	int estimatedChangePoint = -1;
        int N = this.dynamicWindow.size();
        this.cushion = Math.max(100, (int)Math.floor(Math.pow(N, gamma)));
        //mean conf. should not fall below 0.3
        if(N>(2*this.cushion) && calculateMean(0, N-1) <= 0.3)
            return N-1;
        double threshold = -Math.log(this.sensitivity);
        
        double w = 0;
        int kAtMaxW = -1;
        for(int k = this.cushion; k <= N-this.cushion; k++)
        {
            if(calculateMean(k, N-1) <= 0.95*calculateMean(0, k-1))
            {
                double skn = 0;
                /* estimate pre and post change parameters */
                double alphaPreChange = calcBetaDistAlpha(0, k-1);
                double betaPreChange = calculateBetaDistBeta(alphaPreChange, 0, k-1);
                double alphaPostChange = calcBetaDistAlpha(k, N-1);
                double betaPostChange = calculateBetaDistBeta(alphaPostChange, k, N-1);
                
                BetaDistributionImpl preBetaDist = new BetaDistributionImpl(alphaPreChange, betaPreChange);
                BetaDistributionImpl postBetaDist = new BetaDistributionImpl(alphaPostChange, betaPostChange);

                for(int i=k; i<N; i++)
                {
                    try{
                        skn += Math.log(postBetaDist.density(this.dynamicWindow.get(i).doubleValue())/preBetaDist.density(this.dynamicWindow.get(i).doubleValue()));
                    }
                    catch(Exception e){
                        e.printStackTrace();
                        System.out.println("continuing...");
                        skn = 0;
                        break;
                    }
                }        
                if(skn > w)
                {
                    w = skn;
                    kAtMaxW = k;
                }
            }
        }
        if(w >= threshold && kAtMaxW != -1)
        {
            System.out.println("\nChangePoint Found!");
            estimatedChangePoint = kAtMaxW;
            System.out.println("Estimated change point is " + estimatedChangePoint);
        }
        //force change point if confidence falls down terribly
        if(estimatedChangePoint == -1 && N>=100 && calculateMean(0, N-1) < 0.3)
            estimatedChangePoint = N-1;
        return estimatedChangePoint;
    }
    
    /* functions to estimate beta distribution parameters*/
    public double calcBetaDistAlpha(int from, int to)
    {
        double sampleMean = calculateMean(from, to);
        double sampleVariance = calculateVariance(from, to);
        return ((Math.pow(sampleMean, 2) - Math.pow(sampleMean, 3))/sampleVariance)- sampleMean;
    }
    
    public double calculateBetaDistBeta(double alphaPreChange, int from, int to)
    {
        double sampleMean = calculateMean(from, to);
        return alphaPreChange * ((1/sampleMean)-1);
    }
    /*
     * calculate mean of the elements in dynamicWindow
     * both of the indices from and to are inclusive
     */
    public double calculateMean(int from, int to)
    {
    	double sum = 0.0;
    	for(int i=from; i<=to; i++)
    	{
    		sum += this.dynamicWindow.get(i);
    	}
    	return sum/(to-from+1); 
    }
    
    /*
     * calculate mean of the elements in the list
     * both of the indices from and to are inclusive
     */
    public double calculateListMean(ArrayList<Double> list, int from, int to)
    {
    	double sum = 0.0;
    	for(int i=from; i<=to; i++)
    	{
    		sum += list.get(i);
    	}
    	return sum/(to-from+1); 
    }
    
    public double calculateVariance(int from, int to)
    {
        double sumOfSquares = 0.0;
        double mean = calculateMean(from, to);
        for(int i=from; i<=to; i++)
        {
            sumOfSquares += (this.dynamicWindow.get(i) - mean) * (this.dynamicWindow.get(i) - mean);
        }
        return sumOfSquares/(to-from+1); 
    }    
}
