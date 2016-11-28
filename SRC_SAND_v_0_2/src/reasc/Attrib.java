package reasc;

public class Attrib 
{          
          public double[] prob = null;  //probability of different classes
          public double ig = 0;
          
          Attrib(double g, double[] p) 
          { 
              ig = g; 
              prob = p.clone();
          }                                        
  }
