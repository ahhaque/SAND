package mineClass;

import java.util.*;
import weka.core.*;


public class Kmeans {
    
    Minstances m_data = null;   //dataset (with instance ids and cluster ids)     
    public int K = 0;           //#of clusters to build
    int C = 0;                  //#of classes
    int A = 0;                  //#of attributes
    
    public Cluster[] Clusters = null;
    //public int[] ClusterId = null;
    
    Random Generator = new Random(1234567);    
    boolean Changed = false;
            
    /** Creates a new instance of Kmeans */
    public Kmeans(Minstances data, int k) {
        m_data = data;
        K = k;
        if(K == 0)
            K = 1;
        C = data.numClasses();
        A = data.numAttributes();
    }

    //for backward compatibility
    public Kmeans(Instances data, int k) {
        m_data = new Minstances(data, data.numInstances());
        for(int i = 0; i < data.numInstances(); i ++)
            m_data.add(new Minstance(data.instance(i),-1));
        K = k;
        if(K == 0)
            K = 1;
        C = data.numClasses();
        A = data.numAttributes();
    }
    
    //find euclidean distance until cutoff point
    public static double Euclidean(Instance p1, Instance p2, double cutoff)
    {
        int i = 0;
        double d = 0;
        for(i = 0; i < p1.numAttributes(); i ++)
        {
            d = d + (p1.value(i) - p2.value(i)) * (p1.value(i) - p2.value(i));
            if(cutoff >= 0) //then valid cutoff
            {
                if(d >= cutoff)                
                    return d;                
            }
        }
        return d;
    }
    
    //find the most distant point [max(min(p,seed))]    
    int findDistant(int index, int count, boolean[] isSeed, ArrayList data)
    {
        double max = -1, d = 0;
        int mi = -1;
        for(int i = 0; i < data.size(); i ++)
        {
            if(isSeed[i] == true)
                continue;
            
            Instance p = (Minstance)data.get(i);
            
            //first find the min distance from this point to each centroid
            double min = Euclidean(p, Clusters[index].centroid, -1);
            for(int j = index + 1; j < count; j ++)
            {
                d = Euclidean(p, Clusters[j].centroid, min);
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
            
    //Randomly find a seed
    int findRandom(int index, boolean[] isSeed, ArrayList data)
    {
        int mi = -1;
        int indx = (int) Math.random()* data.size();        
                
        for(int i = indx; i < data.size(); i ++)
        {                        
            if(isSeed[i] == false) //found it
            {
                mi = i;                
                break;            
            }
        }
        
        if(mi == -1) //not found, try earlier points
        {
            for(int i = 0; i < index; i ++)
            {                        
                if(isSeed[i] == false) //found it
                {
                    mi = i;                
                    break;            
                }//if
            }//for   
        }//if
                
        return mi;
    }       
    
    int initClusters(int index, int n, ArrayList data)
    {
        if(n >= data.size()) //then one seed for each Minstance
        {
            for(int i = 0; i < data.size(); i ++)
            {
                Minstance x = (Minstance)data.get(i);
                Clusters[index + i] = new Cluster(x, index + i); //create a new cluster        
            }
            return data.size();
        }
        
        boolean[] isSeed = new boolean[data.size()];
        
        int i = (int)(Generator.nextDouble() * data.size()); //randomly select a point
        isSeed[i] = true;
        
        Minstance x = (Minstance)data.get(i);
                        
        Clusters[index] = new Cluster(x, index); //create a new cluster        
        
        int count = 1; //total clusters created
        
        while(count < n)
        {
            i = findDistant(index,count, isSeed, data);  //find the point that is at a 
                                        //maximum distance from each point in the seed
            if(i == -1)                 //could not find any distant point
                return count;
            
            isSeed[i] = true;
            
            x = (Minstance)data.get(i);
                        
            Clusters[index+count] = new Cluster(x, index + count);
            
            count ++;            
        }//while
        
        return count;
    }
            
    //no find-distant; just randomly choose the seeds
    int initClustersNormal(int index, int n, ArrayList data)
    {
        if(n >= data.size()) //then one seed for each Minstance
        {
            for(int i = 0; i < data.size(); i ++)
            {
                Minstance x = (Minstance)data.get(i);
                Clusters[index + i] = new Cluster(x, index + i); //create a new cluster        
            }
            return data.size();
        }
        
        boolean[] isSeed = new boolean[data.size()];
        
        int i = (int)(Generator.nextDouble() * data.size()); //randomly select a point
        isSeed[i] = true;
        
        Minstance x = (Minstance)data.get(i);
                        
        Clusters[index] = new Cluster(x, index); //create a new cluster        
        
        int count = 1; //total clusters created
        
        while(count < n)
        {
            i = findRandom(index, isSeed, data);  //find a random point
            if(i == -1)                 //could not find any distant point
                return count;
            
            isSeed[i] = true;
            
            x = (Minstance)data.get(i);
                        
            Clusters[index+count] = new Cluster(x, index + count);
            
            count ++;            
        }//while
        
        return count;
    }
    
    public void init() //constrained initialization
    {
        double[] Prior = new double[C];
        ArrayList[] subdata = new ArrayList[C];        
                
        int n = m_data.numInstances();
        if(n == 0)
            Constants.logger.debug("Kmeans: data set empty");
        
        for(int i = 0; i < C; i ++)
            subdata[i] = new ArrayList();
        
        for (int i = 0; i < n; i ++)
        {        
            Minstance p = m_data.minstance(i);
            int cls = (int)p.classValue();
            Prior[cls] ++;        
            subdata[cls].add(p);
        }//for
        
        int count = 0;
        
        //first determine #of clusters
        
        int oldk = K;
        K = 0;
        
        for(int i = 0; i < C; i ++)
        {
            if(Prior[i] == 0)
                continue;
            
            int nc = (int) (oldk * Prior[i] / (double) n); //number of seeds for this class
            if(nc == 0)
                nc = 1;
            K += nc;    
        }
                
        Clusters = new Cluster[K];
        
        for(int i = 0; i < C; i ++)
        {
            if(Prior[i] == 0)
                continue;
            
            int nc = (int) (oldk * Prior[i] / (double) n); //number of seeds for this class
            if(nc == 0)
                nc = 1;
            
            //System.out.println("class:"+i + " clusters:" +nc + " sudata size = "+subdata[i].size());            
            count += initClusters(count, nc, subdata[i]);                
            //System.out.println(count+ "  clusters created");            
        }
        //System.out.println("K = "+K+", count = "+count);
        if(K != count)
            System.out.println("ERRorrrrrrr! INIT: not enough clusters");
                       
    }//init()
    
        
    //normal initialization - no constraints (only farthest first heuristic applied)
    public boolean initNormal() 
    {        
        ArrayList data = new ArrayList();
        for(int i = 0; i < m_data.numInstances(); i ++)
            data.add(m_data.minstance(i));

        Clusters = new Cluster[K];
        int count = 0;

        count = initClusters(count, K, data);

        if(K != count)
            System.out.println("ERRorrrrrrr! INIT OUTLIER: not enough clusters");

        return true;
    }
      
    //E step of the EM algorithm
    
    public double E_Step()
    {        
        Changed = false;           
        double globalobj = 0;
        
        for(int i = 0; i < m_data.numInstances(); i ++)
        {           
            Minstance p = m_data.minstance(i);
            double minobj = Euclidean(p,Clusters[0].centroid,-1);
            int minc = 0;
            for(int j = 1; j < K; j ++)
            {
                double mo = Euclidean(p,Clusters[j].centroid,minobj);
                if(mo < minobj)
                {
                    minobj = mo;
                    minc = j;
                }
            }//for j = 1 to K

            if(p.ClusterId != minc) //then cluster changed
            {
                Changed = true;
                p.ClusterId = minc;
            }
            globalobj += minobj;
        }    
        //compute globalobj
                
      return globalobj;
        
    }//E_Step
    
        
    //Re-assign cluster centroids
    public void M_Step()
    {
        //Temporary data points
        
        for(int i = 0; i < K; i ++)
        {                       
            for(int j = 0; j < A; j++)
                Clusters[i].centroid.setValue(j,0);
            Clusters[i].n = 0;
        }
        
        //update avector for each data points
        for(int i = 0; i < m_data.numInstances(); i ++)
        {
            Minstance p = m_data.minstance(i);
            for(int j = 0; j < A; j ++)            
                Clusters[p.ClusterId].centroid.setValue(j,
                Clusters[p.ClusterId].centroid.value(j) + p.value(j));
            Clusters[p.ClusterId].n++;
        }
        
        //take the average
        String debg = "";
        for(int i = 0; i < K; i ++)
        {
            int t = Clusters[i].n;            
            debg +="Cluster"+i+":"+t+",";
            for(int j = 0; j < A; j++)
            {                    
                if(t > 0)
                {                                                       
                    Clusters[i].centroid.setValue(j,
                    Clusters[i].centroid.value(j)/ t);
                }//if
            }//for int j
        }//for int i
        //logger.debug(debg+"\n");
    }//M_step      
    
     public void E_M()
    {
        for(int i = 0; i < K; i ++)
             Clusters[i].init();
        
        //ClusterId = new int[m_data.numInstances()];
        
        int iteration = 0;
        
        long time = System.currentTimeMillis();
        //logger.debug(data.numInstances);
        double obj = E_Step();
        double oldobj = obj + 1;         
        while (Changed)
        {            
            oldobj = obj;
            M_Step();
            obj = E_Step();                        
            //logger.debug("(Old,new) obj = ("+oldobj+","+obj+"),  changed= "+changedAtall);            
            iteration ++;
        }        
        //logger.debug("E_M Iterations done = "+ iteration);
        M_Step();                                      
    }       
    
    public void compStat()
    {        
        //get max, mean, and sd for each cluster                
        for(int i = 0; i < m_data.numInstances(); i ++)
        {
            Minstance p = m_data.minstance(i);
            int cId = p.ClusterId;
            double d = Euclidean(p,Clusters[cId].centroid,-1);
            if(d > Clusters[cId].radius)
                Clusters[cId].radius = d;
            Clusters[cId].sumx += d;
            Clusters[cId].sumx2 += d * d;          
            Clusters[cId].dist[(int)p.classValue()] ++;          
        }
        
        for(int i = 0; i < K; i ++)
        {
            if( Clusters[i].n > 0)
            {
                Clusters[i].meand = Clusters[i].sumx / (double)Clusters[i].n;  //E(x)
                Clusters[i].vard = Clusters[i].sumx2 / (double)Clusters[i].n - //E(x^2) - (E(x))^2
                        Clusters[i].meand * Clusters[i].meand;
              
                Clusters[i].majority = 0;
                for(int j = 1; j < Clusters[i].dist.length; j ++)
                    if(Clusters[i].dist[j] > Clusters[i].dist[(int)Clusters[i].majority])
                        Clusters[i].majority = j;
            }//if n > 0            
        }// for i < K
            
        String debug = "Total clusters= " + K + ", #of data points in clusters: \n";
        for(int i = 0; i < K; i ++)            
            debug += i + ": n =" + Clusters[i].n + " radius="+Clusters[i].radius + " mean=" + Clusters[i].meand + " var ="+Clusters[i].vard + "\n";
        //EvoDT.logger.debug(debug);        
    }
            
    //semi-supervised clustering (only constrained initialization)
    public void mkCluster()
    {
        init();
        E_M();
        compStat();
    }

    //Unsupervised clustering with filtering (remove small clusters)
    public void mkOutlierCluster()
    {
        initNormal();
        E_M();
        compStat();

        //filter out small outlier clusters
        //First, count how many clusters will survive
        int sv = 0;
        for(int i = 0; i < Clusters.length; i ++)
            if(Clusters[i].n > 2)
                sv ++;
        Cluster[] TempClusters = new Cluster[sv];
        int[] newCID = new int[Clusters.length]; //new cluster id
        
        sv = 0;
        for(int i = 0; i < Clusters.length; i ++)
        {
            if(Clusters[i].n > 2)
            {
                TempClusters[sv] = Clusters[i];
                newCID[i] = sv;
                sv ++;
            }
            else
            {
                newCID[i] = -1;                
            }
        }
        //re-assign cluster id
        for(int i = 0; i < m_data.numInstances(); i ++)
            m_data.minstance(i).ClusterId = newCID[m_data.minstance(i).ClusterId];
        
        Clusters = null;
        Clusters = TempClusters;
    }       

    //Unsupervised clustering
    public void mkNormalCluster()
    {        
        initNormal();
        E_M();
        compStat();                
    }
}
