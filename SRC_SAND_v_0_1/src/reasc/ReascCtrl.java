package reasc;

import java.util.*;
import weka.core.*;
import mineClass.Constants;
import mineClass.ConfidenceStats;

public class ReascCtrl extends weka.classifiers.Classifier implements OptionHandler {
    
    
    static int C = 0;           //number of classes    
    int Lp = 0;                 //total number of labeled points in this model
    int Labeledp = 0;           //percentage of labeled data
    int N = 0;                  //total number of points
    int K = 0;                  //number of clusters to build
    double acc = 0.0;           //current accuracy    
    public int Cid = -1;        //classifier unique id
    public double LastError = 0;      //error in the last data chunk
    public boolean IsOutlier = false;
    public double Dist = 0;             //Distance from the perimiter of the nearest cluster
    public double Weight = 0;
    
    /* my added fields for weighted voting */
    public double distFromPer = 0.0; //distance from perimeter
    public double ratioOfLabeledInst = 0; //number of data instances in the closest cluster 
    public double ratioMaxClass = 0.0; //ratio of instances supporting max class in the closest cluster
    /* my added fields for weighted voting */
    
    public Cluster[] Clusters = null;
    public mineClass.Cluster[] mcClusters = null;
    double[] Prior = null;      //prior probabilities of each class
    public boolean[] Dataseen = null;
    Instances Data = null;      //set of instances used to build this model
    
    /* correlation statistics*/
    public ConfidenceStats confidenceStats = new ConfidenceStats();

    void Summary() //delete points
    {
        //which classes are seen?
        Dataseen = new boolean[this.C];
        for(int i = 0; i < Dataseen.length ; i ++)
            Dataseen[i] = false;
        
        N = 0;
        Lp = 0;
        for(int i = 0; i < Clusters.length; i ++)
        {           
            N += Clusters[i].n;
            Lp += Clusters[i].lp;
            if(Clusters[i].majority != Constants.NIL)
                Dataseen[Clusters[i].majority] = true;
        }        
    }
                            
    /** Creates a new instance of ReascCtrl */
    public void init(int mno, Cluster[] c, double[] prior, int lp) {
        
        //find number of labeled clusters
        int lb = 0;
        for(int i = 0; i < c.length; i ++)
            if(c[i].majority >= 0) //then labeled
                lb ++;
        
        //Save only the labeled clusters
        Clusters = new Cluster[lb];
        int j = 0;
        for(int i = 0; i < c.length; i ++)
        {
            if(c[i].majority >= 0)
                Clusters[j ++] = new Cluster(c[i]);
        }
        
        Lp = lp;
        //Prior = new double[ReascCore.C];
        Prior = prior.clone();
               
        Summary();     
        
    }//public model

    public void buildClassifier(Instances data)
    {
        //first convert Instance to Datapoint
        if(this.C == 0)
            this.C = data.numClasses();
        ReascCore ssc = new ReascCore(this.K, this.Labeledp, this.C);
        this.Data = data;
        ssc.Data = new ArrayList();
        for(int i = 0; i < data.numInstances();  i ++)
        {
            Instance ins = data.instance(i);
            Datapoint p = new Datapoint(ins.toDoubleArray(),(int)ins.classValue(),-1,false,0);
            ssc.Data.add(p);
        }

        //now build the clusters
        ssc.init();
        ssc.E_M();
        init(0, ssc.Clusters, ssc.Prior, ssc.Lp);
        mcClusters = this.getClusters();
        /*Constants.logger.debug("Clusters............");
        for(int i = 0; i < mcClusters.length; i ++)
        {
            Constants.logger.debug("Cluster #" + i + " Frequencies: " +
                    mcClusters[i].dist[0]+","+mcClusters[i].dist[1] + " Majority = " + mcClusters[i].majority);
        }*/

        //return m;
    }
    
    
   //P(datapoint is in class cls | model s)
    @Override
    public double classifyInstance(Instance ins)
    {
        //get the closest labeled centroid                
        //Heap h = new Heap(Q);                
       
        Datapoint p = new Datapoint(ins.toDoubleArray(),(int)ins.classValue(),-1,false,0);
        //get the Q nearest neighbors of p
        
        double[] prob = new double[ReascCore.C];
        double mind = 1E50;
        int mini = -1;

        //String debug = "Distances:";
        
        //for each cluster
        for(int c = 0; c < this.Clusters.length; c ++)
        {
            if(this.Clusters[c].lp == 0)
                continue;
            
            double d = Datapoint.Euclidean(p, this.Clusters[c].centroid);
            //debug += " " + d + " ";
            if(mind > d)
            {
                mind = d;
                mini = c;
            }
        }
        //Constants.logger.debug(debug);
        int maxc = -1;
        ArrayList<Double> classWiseVotes = new ArrayList<Double>();
        
        for(int k = 0; k < ReascCore.C; k ++)
        {
            if(this.Clusters[mini].lp > 0)
            {
                prob[k] = (double)this.Clusters[mini].frq[k] / (double)this.Clusters[mini].lp;
                classWiseVotes.add(prob[k]);
            }
            else
            {
                prob[k] = 0;
                classWiseVotes.add(prob[k]);
            }
            if(maxc == -1)
                maxc = k;
            else if (prob[maxc] < prob[k])
                maxc = k;
        }
        Collections.sort(classWiseVotes);
        
        this.distFromPer = this.Clusters[mini].radius - Datapoint.Euclidean(p, this.Clusters[mini].centroid); //negative value means out of the cluster area
        this.ratioOfLabeledInst = this.Clusters[mini].lp/this.Lp; //don't use this criteria currently 
        this.ratioMaxClass = (double)this.Clusters[mini].frq[maxc]/(double)this.Clusters[mini].lp;
        //h.insert(d, prob);            
        this.IsOutlier = false;
        if(mini != -1)    
        {
            if(mind > this.Clusters[mini].radius) //outside the boundary
            {
                this.IsOutlier = true;
                this.Dist = mind - this.Clusters[mini].radius;
            }
        }
        //Constants.logger.debug("Instance class=" + (int)ins.classValue() + " Prediction = " + maxc + " Outlier = "+ this.IsOutlier + " Closest cluster = " + mini + "  Cluster majority = " + this.Clusters[mini].majority);
        return maxc;
    }
    
    
     //merge two closest clusters in m
    void Merge(ReascCtrl m)
    {
        //find the closest pair
        double min = -1.0;
        int mi = -1,mj = -1;
        
        for(int i = 0; i < m.Clusters.length - 1; i ++)
        {
            for(int j = i + 1; j < m.Clusters.length; j ++)
            {
                if(m.Clusters[i].majority == m.Clusters[j].majority)
                {
                    double d = Datapoint.Euclidean(m.Clusters[i].centroid,m.Clusters[j].centroid);
                    if((min < 0) || (d < min))
                    {
                        min = d;
                        mi = i;
                        mj = j;
                    }//if                    
                }//if majority
            }//for j
        }//for i
        
        //now mi and mj are the closest clusters
        if(mi == -1 || mj == -1)
            System.out.println("Error: no closest pair of clusters found");
        else
        {
            //merge mi and mj
            
            
            //Compute new centroid
            for(int i = 0; i < m.Clusters[mi].centroid.avector.length; i ++)
            {
               m.Clusters[mi].centroid.avector[i] = (m.Clusters[mi].centroid.avector[i] * m.Clusters[mi].n +
                                              m.Clusters[mj].centroid.avector[i] * m.Clusters[mj].n) / (m.Clusters[mi].n + m.Clusters[mj].n);
            }
            double d = Datapoint.Euclidean(m.Clusters[mi].centroid, m.Clusters[mj].centroid); //distance between two centroids
            m.Clusters[mi].radius = (d + m.Clusters[mi].radius + m.Clusters[mj].radius)/2.0; //new radius
            
            for(int i = 0; i < m.Clusters[mi].frq.length; i ++)
                m.Clusters[mi].frq[i] += m.Clusters[mj].frq[i];
            
            for(int i = 0; i < m.Clusters[mi].sum.length; i ++)
                m.Clusters[mi].sum[i] += m.Clusters[mj].sum[i];
            
            m.Clusters[mi].n += m.Clusters[mj].n;
            m.Clusters[mi].lp += m.Clusters[mj].lp;
            
            //Replace mj with the last cluster
            m.Clusters[mj] = m.Clusters[m.Clusters.length - 1];
        }
    }
    
        
    //for inheritance
    public String getRevision()
    {
        return ReascCore.Version;
    }

   /*OPTION HANDLER METHODS*/
   /*
   * Returns an enumeration describing the available options.
   *
   * Valid options are: <p>
   *
   * -K clusters <br>
   * Number of clusters per model (default 50).<p>
   *  
   * -P labeled_percentage <br>
   * Perentage of labeled data (default 95) <p>
   *
   * -C classes <br>
   * Total number of classes (default 2) <p>
   *
   
   * @return an enumeration of all the available options.
   */
   
    @Override
  public Enumeration listOptions() {

    Vector newVector = new Vector(3);

    newVector.
	addElement(new Option("\tSet the number of micro-clusters per model." +
                  "\t(default 50)",
			      "K", 1, "-K <number of micro-clusters>"));
    newVector.
	addElement(new Option("\tSet the percentage of labeled data." +
			      "\t(default 95)",
			      "P", 1, "-P <perc labeled>"));    
    
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
    @Override
  public void setOptions(String[] options) throws Exception {

    // Other options
    String numClusters = Utils.getOption('K', options);
    if (numClusters.length() != 0) {
      this.K = Integer.parseInt(numClusters);
    } else {
      this.K = 50;
    }
    
    String percString = Utils.getOption('P', options);
    if (percString.length() != 0) {
      this.Labeledp = Integer.parseInt(percString);
	  }
     else {
      this.Labeledp = 95;
    }   
    
  } //setOptions

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
    @Override
  public String [] getOptions() {

    String [] options = new String [3];
    int current = 0;
    
	options[current++] = "-K"; options[current++] = "" + this.K;
	options[current++] = "-P"; options[current++] = "" + this.Labeledp;
    options[current++] = "-C"; options[current++] = "" + this.C;

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  public mineClass.Cluster[] getClusters()
  {
        //reasc.Cluster[] Clusters = ((reasc.ReascCtrl)en[i]).Clusters;
        //convert to evodt.Cluster
        mineClass.Cluster[] clusters = new mineClass.Cluster[this.Clusters.length];
        for(int j = 0; j < clusters.length; j ++)
        {
            Instance centroid = new Instance(1,Clusters[j].centroid.avector);
            Instance sum = new Instance(1,Clusters[j].sum);
            centroid.setDataset(this.Data);
            sum.setDataset(this.Data);

            clusters[j] = new mineClass.Cluster(centroid,j);
            //clusters[j].sum = new Instance(sum);
            if(Clusters[j].frq != null)
                clusters[j].dist = Clusters[j].frq.clone();
            clusters[j].n = Clusters[j].n;
            clusters[j].radius = Clusters[j].radius;
            clusters[j].majority = Clusters[j].majority;
        }

        return clusters;
  }
    
    
}//class model
