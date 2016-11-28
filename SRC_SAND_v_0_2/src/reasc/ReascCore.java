package reasc;

import java.util.*;
import org.apache.log4j.*;
import mineClass.Constants;


public class ReascCore {
    static String Version = "27.0";
    ArrayList Data = new ArrayList(); //all training data points
    ArrayList TestData = new ArrayList(); //test data points
    
    static int C = 0;           //total number of classes
    ArrayList[] Ldata = null;   //labeled data (one arraylist for each type)
    ArrayList Udata = null;     //Unlabeled data
    double[] Prior = null;      //Prior probabilities of each class    
    int Lp = 0;                 //total number of labeled points
    
    boolean[] Dataseen = null;  //Has the data of class i been seen (in any of the models)?
    int[] Datapresent = null;   //In how many models is the data of a class present now? 
    boolean Updatemodel = false;//Do we need to update previous models?
    boolean[] Newclass = null; //Is this class new to the models?
        
    int Kmin = 0;   //minm #of clusters
    public int K = 0;      //#of clusters
    //public int M = 0;      //#of saved models
    //public int Q = 0;      //#Q of Q-NN
        
    Cluster[] Clusters = null;    
    ReascCtrl[] Models = null;
    //ReascCtrl LastModel = null;
    
    //int S = 0; //total snapshots saved so far
    int BestModel = 0, WorstModel = 0;
    double Labeledp = 0; //percentage of labeled points        
    
    int Pos = 0; //total positive
    int Neg = 0; //total negative
    int FP = 0; //total false positives
    int FN = 0; //total false negatives
    int Err = 0; //total error
    int Tpts = 0; //total number of points
    int Tem = 0;//total number of E-M iterations done
    int Ticm = 0;   //total number of ICM iterations
    long Tmem = 0; //total running time of E-M process
    long Tmtrain = 0;   //total training time
    long Tmtest = 0; //total testing time
    
    Random Generator = new Random(Constants.Seed);
    
    //logger for log4j
    static Logger  logger = Logger.getLogger(ReascCore.class);
    
    boolean changedAtall = false; //any cluster changes after E_M?
    static long ID = 0; //unique id of data points
    
    double Dmax = 0; //maximum distance betweeen two points
    long Time = 0;
    
    boolean Entropy = true; //use entropy as impurity measure (if false then use GINI index)
    
    public ReascCore(int k, int lp, int numClasses)
    {
        this.K = k;
        //this.Q = q;
        //this.M = m;
        this.Labeledp = lp;
        //this.Entropy = entropy;
        this.C = numClasses;
    }
    
    //find the most distant point [max(min(p,seed))]
    
    int findDistant(int index, int count, boolean[] isSeed, ArrayList data)
    {
        double max = 0, d = 0;
        int mi = 0;
        for(int i = 0; i < data.size(); i ++)
        {
            if(isSeed[i] == true)
                continue;
            Datapoint p = (Datapoint)data.get(i);
            
            //first find the min distance from this point to each centroid
            double min = Datapoint.Euclidean(p, Clusters[index].centroid);
            for(int j = index + 1; j < count; j ++)
            {
                d = Datapoint.Euclidean(p, Clusters[j].centroid);
                if(d < min)
                    min = d;
            }
            
            //now check if this min is the max
            if(min > max)
            {
                max = min;
                mi = i;
            }
        }
        return mi;
    }       
        
    void initClusters(int index, int n, ArrayList data)
    {
        if(n == data.size()) //then one seed for each datapoint
        {
            for(int i = 0; i < n; i ++)
            {
                Datapoint x = (Datapoint)data.get(i);                                
                Clusters[index + i] = new Cluster(x, index + i); //create a new cluster        
            }
            return;
        }
        
        boolean[] isSeed = new boolean[data.size()];
        
        int i = (int)(Generator.nextDouble() * data.size()); //randomly select a point
        isSeed[i] = true;
        
        Datapoint x = (Datapoint)data.get(i);
                        
        Clusters[index] = new Cluster(x, index); //create a new cluster        
        
        int count = 1; //total clusters created
        
        while(count < n)
        {
            i = findDistant(index,count, isSeed, data);  //find the point that is at a 
                                               //maximum distance from each point in the seed
            isSeed[i] = true;
            
            x = (Datapoint)data.get(i);
                        
            Clusters[index+count] = new Cluster(x, index + count);
            count ++;            
        }//while
    }
    
    //return maximum prior probability
    int maxPrior()
    {
        int max = 0;
        for(int i = 1; i < C; i ++)
            if(Prior[i] > Prior[max])
                max = i;
        
        return max;
    }
   
    
    void label() //label some points
    {
        //logger.debug("C = "+C);
        Ldata = new ArrayList[C];
        Udata = new ArrayList(); 
        Prior = new double[C];
        if(Dataseen == null)
        {
            Dataseen = new boolean[C];            
        }
        if(Datapresent == null)
        {
            Datapresent = new int[C];            
        }        
        if(Newclass == null)
        {
            Newclass = new boolean[C];            
        }
        
        
        for(int i = 0; i < C; i ++)
            Dataseen[i] = false;
        
        for(int i = 0; i < C; i ++)
        {
            Ldata[i] = new ArrayList();
            Prior[i] = 0;
        }        
        Lp = 0;
        
        double l = Labeledp;
        Updatemodel = false;        
        
        //label points
        for (int i = 0; i < Data.size(); i ++)
        {        
            if(Generator.nextDouble() <= l/100.0)
            {
                Datapoint p = (Datapoint)Data.get(i);            
                p.labeled = true;           //p is now labeled
                p.classId = p.trueId;       //set the classid                                
                Ldata[p.classId].add(new Datapoint(p));                
                ++Lp;
                Prior[p.classId] ++;
                Dataseen[p.classId] = true;
            }//if
            else    //unlabeled data
            {
                Udata.add((Datapoint)Data.get(i));
            }
        }//for
        
        //set prior probabilites
        for(int i = 0; i < C; i ++)
        {
            //logger.debug("Before: Prior["+i+"]="+Prior[i] + " Lp="+Lp);
            if(Lp > 0)
                Prior[i] = Prior[i] / Lp;
            //logger.debug("After: Prior["+i+"]="+Prior[i] + " Lp="+Lp);
        }
                
    }//label
    
    public int nonZeroPriors(double[] Prior)
    {
        int nzp = 0;
        for(int i=0; i<Prior.length; i++)
        {
            if(Math.abs(Prior[i]-0.0)>0.00001)
            {
                nzp++;
            }
        }
        return nzp;
    }
    
    public void init() //initialization
    {
        //assuming that data have been loaded into "Data"
        //randomly assign labels to Labeledp percent points 
         label();
         
        //logger.debug("(cp,cn)="+Cp + ", "+Cn+"\n");
        //logger.debug("Data labeled ,Lp=" + Lp);
                         
        int count = 0;
        //the following two are to find the minimum prior
        int minPriorIndex = -1;
        double minPrior = 1.0;
        /*at least we need to create one clusters for every classes that
        has data instances in this training data set */
        K = (int)Math.max(K, nonZeroPriors(Prior));
        Clusters = new Cluster[K];
        for(int i = 0; i < C; i ++)
        {
            //this if is to find the minimum prior
            if(Prior[i]>0.0 && minPrior > Prior[i])
            {
                minPrior = Prior[i];
                minPriorIndex = i;
            }
            //create clusters
            int nc = (int)(K * Prior[i]); //number of seeds for this class
            //logger.debug("Prior["+i+"]="+Prior[i]);
            if(nc > 0)
            {
                //logger.debug("class:"+i + " clusters:" +nc);
                initClusters(count, Math.min(nc, Ldata[i].size()), Ldata[i]);
                count += Math.min(nc, Ldata[i].size());
                //logger.debug(count+ "clusters created");
            }
        }
        //if more seeds are needed take from class having min instances
        if(count < K)
        {
            if(!Udata.isEmpty())
                initClusters(count, K - count, Udata);
            else
            {
                initClusters(count, K - count, Ldata[minPriorIndex]);
            }
        }                  
    }//init()
    
  
    double Objective(Datapoint p, int c)
    {
        double v = Datapoint.Euclidean(p,Clusters[c].centroid);
        double en = 0;
        double otherCls = 0; 
        if(p.labeled == true) //then entropy will change
        {
            double[] pi = new double[Clusters[c].frq.length];
            
            //compute pi's
            for(int i = 0; i < Clusters[c].frq.length; i ++)
            {
                if(p.classId == i)
                {
                    //logger.debug(Clusters[c].lp);
                    pi[i] = (Clusters[c].frq[i] + 1) / (Clusters[c].lp + 1);                    
                }
                else //
                {
                    pi[i] = (Clusters[c].frq[i]) / (Clusters[c].lp + 1);
                    otherCls += Clusters[c].frq[i];
                }
            }
                                   
            //compute entropy
            if(Entropy == true)
            {
                for(int i = 0; i < pi.length; i ++)
                {
                    if(pi[i] > 0)
                        en -= pi[i] * (Math.log10(pi[i])/Math.log10(2.0));
                }
            }
            else //use GINI index
            {
                for(int i = 0; i < pi.length; i ++)
                {
                    if(pi[i] > 0)
                        en += pi[i] * pi[i];
                }
                en = 1 - en;
            }                
        }
        return v * (1 + en * otherCls);
    }
    
    //E step of the EM algorithm
    //Implements iterative conditional modes (ICM )
    public double E_Step()
    {
        boolean changed = true;
        changedAtall = false;
        
        double globalobj = 0; //global objective function
        int iteration = 0;
           
        while(changed)
        {
            double oldobj = globalobj;
            globalobj = 0;
            //generate random numbers
            int[] order = new int[Data.size()];       //random order            
            for(int i = 0; i < Data.size(); i ++)
                order[i] = i;
            for(int i = 0; i < Data.size(); i ++)    //generate a random number
            {
                int r = (int) (Generator.nextDouble() * Data.size());
                int t = order[r];                   //swap with the first element
                order[r] = order[0];
                order[0] = t;
            }
            changed = false;           
            for(int i = 0; i < Data.size(); i ++)
            {
                Datapoint p = (Datapoint)Data.get(order[i]); //now pick up a data point in random order                
                
                //Assign the point to the cluster that minimizes the objective function
                
                //first remove the point from the existing cluster
                
                if(p.clusterId != Constants.NIL)
                    Clusters[p.clusterId].removepoint(p);
                        
                double minobj = Objective(p,0);
                int minc = 0;
                for(int j = 1; j < K; j ++)
                {
                    double mo = Objective(p,j);
                    if(mo < minobj)
                    {
                        minobj = mo;
                        minc = j;
                    }
                }//for j = 1 to K
                
                if(p.clusterId != minc) //then cluster changed
                {
                    changed = true;
                    changedAtall = true;
                    logger.trace("cluster changed.......");
                    
                    //remove p from old cluster
                    if(p.clusterId != Constants.NIL)  //then it is already assigned
                    {
                        logger.trace("delete (c,Id)="+p.clusterId+","+p.Id);
                        //Clusters[p.clusterId].removepoint(p);                    
                    }                                        
                }
                
                //add to the new cluster
                p.clusterId = minc;
                Clusters[minc].addpoint(p);  
                logger.trace("add (c,Id)="+minc+","+p.Id);
                                                
            }//for i = 0 to Data.size()                                      
            iteration ++;
            /*globalobj = 0;
            for(int i = 0; i < Data.size(); i ++)
            {
                globalobj += Objective((Datapoint)Data.get(i),((Datapoint)Data.get(i)).clusterId);
            }
           logger.debug(iteration + " "+globalobj);*/
        }//while changed  
        //compute globalobj
        globalobj = 0;
        for(int i = 0; i < Data.size(); i ++)
        {
            globalobj += Objective((Datapoint)Data.get(i),((Datapoint)Data.get(i)).clusterId);
        }
        //logger.debug("Total ICM iterations done = "+iteration + "Objective = "+globalobj);
        Ticm += iteration;
        return globalobj;
        //return changedAtall;
    }//E_Step
    
        
    //Re-assign cluster centroids
    public void M_Step()
    {
        //Temporary data points
        Datapoint[] tmp = new Datapoint[K];
        double[] av = new double[((Datapoint)Data.get(0)).avector.length];
        
        //initialize        
        for(int i = 0; i < K; i ++)
        {            
            tmp[i] = new Datapoint(av,"",Constants.NIL,false,Constants.NIL);
            for(int j = 0; j < tmp[i].avector.length; j++)
                tmp[i].avector[j] = 0;
        }
        
        //update avector for each data points
        for(int i = 0; i < Data.size(); i ++)
        {
            Datapoint p = (Datapoint)Data.get(i);
            for(int j = 0; j < p.avector.length; j ++)
                tmp[p.clusterId].avector[j] += p.avector[j];
        }
        
        //take the average
        String debg = "";
        for(int i = 0; i < K; i ++)
        {
            Clusters[i].compfreq();
            int t = Clusters[i].n;
            debg +="Cluster"+i+":"+t+",";
            for(int j = 0; j < Clusters[i].centroid.avector.length; j++)
            {                                
                if(t > 0)
                {
                    Clusters[i].centroid.avector[j] = tmp[i].avector[j] / t;
                }//if
            }//for int j
        }//for int i
        //logger.debug(debg+"\n");
    }//M_step      
    
    public void E_M()
    {
        for(int i = 0; i < K; i ++)
             Clusters[i].init();
        int iteration = 0;
        
        long time = System.currentTimeMillis();
        //logger.debug(Data.size());
        double obj = E_Step();
        //boolean changed = E_Step();
        double oldobj = obj + 1;
        //while (Math.abs(oldobj - obj) > 0.01)
        while (changedAtall)
        {            
            oldobj = obj;
            M_Step();
            obj = E_Step();            
            //changed = E_Step();
            //logger.debug("(Old,new) obj = ("+oldobj+","+obj+"),  changed= "+changedAtall);            
            iteration ++;
        }        
        //logger.debug("E_M Iterations done = "+ iteration);
        M_Step();        
        Tem += (iteration + 1);
        Tmem += System.currentTimeMillis() - time;
        
        //compute statistics for each cluster
        for(int i = 0; i < Clusters.length; i ++)
        {
            Clusters[i].compfreq();       
            Clusters[i].radius = 0;
        }
        
        //compute the readius of each cluster
        for(int i = 0; i < Data.size(); i ++)
        {
            Datapoint p = (Datapoint)Data.get(i);
            double d = Datapoint.Euclidean(p, Clusters[p.clusterId].centroid);
            if( d > Clusters[p.clusterId].radius)
                Clusters[p.clusterId].radius = d;
        }
    }
   
   
}