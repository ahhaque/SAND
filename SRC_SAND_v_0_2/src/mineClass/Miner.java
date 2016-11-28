/*
 *   Next versions of this project will be documented in root.VersionInfo.java
 */
package mineClass;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.classifiers.Classifier;
import org.apache.log4j.*;
import change_point_detection.BetaDistributionChangePoint;

public class Miner implements OptionHandler{
    
    /** Creates a new instance of Miner */
    public Miner() {
    }

    String InFile = "";     //input file name
    public static BufferedWriter Allinst = null;      //All instance's output
    public static int K = 0;//#of clusters
    public int M = 0;       //ensemble size
    public int Minpts = 0;  //Parameter Minpts
    int ChunkSize = 0;      //Chunk size
    int Tl = 0;             //Labeling delay (exact)
    int Tc = 0;             //Classification delay (maximum allowable)
    String ClassifierName = ""; //name of the classifier to be used
    boolean ShowHelp = false;   //

    public static Logger  logger = Logger.getLogger(Constants.class);
    int TempError = 0;   //temporary error: last batch of T instances
    int TempFP = 0;         //temporary false positive (new class declared but not new class)
    int TempFN = 0;         //temporary true positive (new class correctly detected)
    int TempNC = 0;         //temporary false negative (new class could not be detected)
      
    boolean ClassAppeared[] = null; //which classes have already appeared in the training data?    
    boolean[] TheNewClass = null;   //new classes that appeared
    
    int LastCheck = 0;              //When novel class was checked last time
    int GlobalError = 0;            //Error per chunk (my method)
    int GlobalFP = 0;               //FP per chunk (my method)
    int GlobalFN = 0;               //FN per chunk (my method)
    int GlobalNC = 0;               //New class per chunk (my method)
  
    long TrainTime = 0;             //Total training time
    long TestTime = 0;              //Total test time 
    long CurTime = 0;               //Current time

    ResultPerK Results = new ResultPerK();

    //Test from the scratch
    double testScratch(Classifier C, Instances data)
    {        
        double[] P = new double[data.instance(0).numClasses()];        
        double err = 0;
        for(int i = 0; i < data.numInstances(); i ++)
        {
            try{
                    //Datapoint p = new Datapoint(data.instance(i).toDoubleArray(),(int)data.instance(i))
                    if(C.classifyInstance(data.instance(i)) != data.instance(i).classValue())
                    {
                        err ++;
                    }//if
                    P[(int)data.instance(i).classValue()] ++;
            }
            catch(Exception e)
            {
                System.out.println(e.getMessage());
            }
        }//for
        double MSErr = 0;        
        for(int i = 0; i < P.length; i ++)
        {
            P[i] /= (double)data.numInstances();
            MSErr += P[i] * (1-P[i]) * (1 - P[i]);
        }
        
        return MSErr - err / (double)data.numInstances();
    }//double

    //retrieve the test results for individual classifiers (if not found - do classify)
    double test(Classifier C, Minstances data)
    {
        double[] P = new double[data.instance(0).numClasses()];
        double err = 0;
        for(int i = 0; i < data.numInstances(); i ++)
        {
            Minstance p = data.minstance(i);
            boolean correct = false;
            try{
                    MapPrediction ret = p.getPrediction(C.getClass().getDeclaredField("Cid").getInt(C));
                    if(ret == null) //the classifier did not test this instance (newer classifier)
                    {
                        double rv = C.classifyInstance(p);
                        boolean outlier = C.getClass().getDeclaredField("IsOutlier").getBoolean(C); //also outlier
                        double dist = C.getClass().getDeclaredField("Dist").getDouble(C);
                        ret = new MapPrediction(C.getClass().getDeclaredField("Cid").getInt(C),rv,outlier,dist);
                        p.addPrediction(ret);
                    }
                    if(ret.Predclass == p.classValue()) //correct prediction
                    {
                        correct = true;
                    }//if
                    else //incorrect prediction
                    {
                       if(ret.Isoutlier) // then outlier declared
                       {
                          //if(p.EPrediction.Isoutlier) //Foutlier /* DOUBT */
                          //{
                               if( ( (boolean[]) C.getClass().getDeclaredField("Dataseen").get(C))[(int)p.classValue()] == false)
                                   correct = true; //really a novel class for this classifier, but detected correctly
                          //}//if epred
                       }//if outlier
                    }//incorrect
                    if(!correct)
                        err ++;

                    P[(int)data.instance(i).classValue()] ++;
            }
            catch(Exception e)
            {
                System.out.println(e.getMessage());
            }
        }//for
        double MSErr = 0;
        for(int i = 0; i < P.length; i ++)
        {
            P[i] /= (double)data.numInstances();
            MSErr += P[i] * (1-P[i]) * (1 - P[i]);
        }

        double weight = MSErr - err / (double)data.numInstances();

        try{
            C.getClass().getDeclaredField("Weight").setDouble(C,weight);
        }
        catch(Exception e)
        {
            System.out.println(e.getMessage());
        }

        return weight;
    }

    //Classify a single instance using the ensemble

    int classifySingle(Classifier[] en, int es, Minstance inst) throws Exception
    {
        //recalculate ensemble prediction and outlier status
        //logger.debug("ClassifySingle.."+"ID="+inst.Id);
        double[] predClass = new double[inst.numClasses()];
        
        //boolean foutlier = false;
        int foutlier = 0;
        int notoutlier = 0;
        double outlmxwt = 0;

        for(int j = 0; j < predClass.length; j ++)
            predClass[j] = 0;
        //foutlier = true;
                
        /* for each model, we have two different criteria
        and one final non normalized confidence, total three fields
        */
        double[][] confVals = new double[es][3]; 
        double[] ret = new double[es];
        for(int j=0; j<es; j++)
            ret[j] = -1;
        double minConf = Double.MAX_VALUE;
        double maxConf = Double.MIN_VALUE;
        for(int j = 0; j < es; j ++)
        {
            MapPrediction pred = inst.getPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j]));
            //String debug = "";
            //debug += "Classifier id = " + en[j].getClass().getDeclaredField("Cid").getInt(en[j]);
            
            if(pred == null)
            {
                //System.out.println("ERROR - PREDICTION IS NULL");
                //debug += "  pred = null";
                try
                {
                    ret[j] = en[j].classifyInstance(inst);    //test using the classifier
                }

                catch(Exception e)
                {
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }

                if(ret[j] < 0)
                        logger.debug("return value negative");

                //debug += " outlier? =" + isOutlier + " dist = " + dist + " ret=" + ret;
        
                /* lets calculate its weight if it is not the warming up period*/
                /* calculate weight of this classifier, w_i^x = h_i^x . r_i */
                /* inst.Id == -1 means it is in initial warm up period */
                if(inst.Id != -1) 
                {
                    ConfidenceStats confStats = (ConfidenceStats)en[j].getClass().getDeclaredField("confidenceStats").get(en[j]);

                    if(Math.abs(confStats.getMaxConfVals().get(0) - confStats.getMinConfVals().get(0)) > 0.00001)
                    {
                       confVals[j][0] = (en[j].getClass().getDeclaredField("distFromPer").getDouble(en[j]) - confStats.getMinConfVals().get(0))/(confStats.getMaxConfVals().get(0) - confStats.getMinConfVals().get(0)); 
                    }
                    else //confVals[j][0] == 0
                    {
                        confVals[j][0] = (en[j].getClass().getDeclaredField("distFromPer").getDouble(en[j]) - confStats.getMinConfVals().get(0))/0.00001;
                    }

                    if(Math.abs(confStats.getMaxConfVals().get(1) - confStats.getMinConfVals().get(1)) > 0.00001)
                    {
                        confVals[j][1] = (en[j].getClass().getDeclaredField("ratioMaxClass").getDouble(en[j]) - confStats.getMinConfVals().get(1))/(confStats.getMaxConfVals().get(1) - confStats.getMinConfVals().get(1));
                    }
                    else
                    {
                        confVals[j][1] = (en[j].getClass().getDeclaredField("ratioMaxClass").getDouble(en[j]) - confStats.getMinConfVals().get(1))/0.00001;
                    }

                    confVals[j][2] = (confVals[j][0] * confStats.getCorrCoeff().get(0)) + (confVals[j][1] * confStats.getCorrCoeff().get(1));
                    if(confVals[j][2] > maxConf)
                        maxConf = confVals[j][2];
                    if(confVals[j][2] < minConf)
                        minConf = confVals[j][2];
                }
            }
        }//for j (all classifiers)

        /* now normalize confidences for each model */
        for(int j=0; j<es; j++)
        {
            MapPrediction pred = inst.getPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j]));
            if(pred == null)
            {
                //check if the test data is a raw outlier
                boolean isOutlier = en[j].getClass().getDeclaredField("IsOutlier").getBoolean(en[j]);
                double dist = en[j].getClass().getDeclaredField("Dist").getDouble(en[j]);
                
                if(inst.Id != -1)
                {
                    if(Math.abs(maxConf-minConf) > 0.00001)
                    {
                        confVals[j][2] = (confVals[j][2]-minConf)/(maxConf-minConf);    
                    }
                    else
                    {
                        confVals[j][2] = (confVals[j][2]-minConf)/0.00001;
                    }
                    ConfidenceStats confStats = (ConfidenceStats)en[j].getClass().getDeclaredField("confidenceStats").get(en[j]);

                    //special treatment if accuracy == 1
                    if(confVals[j][0] > 0 && Math.abs(confStats.accuracy - 1.0) < 0.00001) 
                        confVals[j][2] = 1.0;
                    
                    inst.addPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j]), ret[j], isOutlier, dist, confVals[j][0], confVals[j][1], confVals[j][2]);                
                }
                else
                {
                    inst.addPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j]), ret[j], isOutlier, dist, 
                            en[j].getClass().getDeclaredField("distFromPer").getDouble(en[j]), en[j].getClass().getDeclaredField("ratioMaxClass").getDouble(en[j]), -1);
                }
                pred = inst.getPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j]));
            }
            
            boolean noClass = false;
            //see if any other classifier voted for a class that this classifier does not know of
            for(int k = 0; k <inst.numClasses(); k ++)
            {
                if(predClass[k] > 0 &&
                ((boolean[])en[j].getClass().getDeclaredField("Dataseen").get(en[j]))[k] == false)
                {
                    noClass = true;
                    break;
                }
            }
            double wt = Math.exp(-pred.Dist);
            if(noClass == true&& pred.Isoutlier || wt < Constants.OUTTH && pred.Isoutlier) //then do not count its vote
                ;//logger.debug("no vote"); //no vote
            else
            {
                //logger.debug("count vote");
                //if conf is used towards voting
                if(pred.Isoutlier == false && Constants.USECONFTOWARDSVOTING == true)
                {
                    if(inst.Id == -1)
                        predClass[(int)pred.Predclass] ++;
                    else
                        predClass[(int)pred.Predclass] += pred.weight;
                }
                //if conf is not used towards voting
                else if(pred.Isoutlier == false && Constants.USECONFTOWARDSVOTING == false)
                {
                    predClass[(int)pred.Predclass] ++;
                }
                //if pred.Isoutlier is true
                else
                {                   
                    predClass[(int)pred.Predclass] += wt;
                    //logger.debug("Weight added = " + wt);
                }
                //logger.debug("class vote for class "+(int)pred.Predclass +"="+ predClass[(int)pred.Predclass]);
            }
            //if(pred.Isoutlier == false)
            //    foutlier = false; //not an outlier by this classifier
            if(pred.Isoutlier)
                foutlier ++;
            else
                notoutlier ++;

            if(wt > outlmxwt)
                outlmxwt = wt;
        }

        int votedClass = 0; //max
        
        //Uniform-weight voting
        for(int j = 1; j < predClass.length; j ++)
        {
            if(predClass[j] > predClass[votedClass])
                votedClass = j;
        }

        inst.EPrediction = new MapPrediction(-1, votedClass, notoutlier == 0 && outlmxwt < Constants.OUTTH, outlmxwt);
        //logger.debug(" Eprediction.predclass=" + inst.EPrediction.Predclass);

        return votedClass;
    }

    int filterOutliers(Classifier[] en, int es, Minstances outlierList, int curTime) throws Exception
    {
        int totUnComitted = 0; //how many uncommited were filtered?
        //logger.debug("Filtering outliers..");
        
        for(int i = 0; i < outlierList.numInstances(); i ++)
        {
            Minstance outli = outlierList.minstance(i);

            if(outli.Id + this.Tl <= curTime
                    && ClassAppeared[(int)outli.classValue()] == true) //labeled and existing class
            {
                 outlierList.delete(i);
                --i;
                continue;
            }

            boolean needfilter = false;

            //see if outli needs filtering
            for(int j = 0; j < es; j ++)
            {
                if(outli.getPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j])) == null)
                {
                    needfilter = true; //need to classify with this classifier
                    double ret = en[j].classifyInstance(outli);
                    boolean isOutlier = en[j].getClass().getDeclaredField("IsOutlier").getBoolean(en[j]);
                    double dist = en[j].getClass().getDeclaredField("Dist").getDouble(en[j]);
                    outli.addPrediction(en[j].getClass().getDeclaredField("Cid").getInt(en[j]), ret, isOutlier, dist);
                }//if
            }//for j

            classifySingle(en, es, outli);

            if(outli.EPrediction.Isoutlier == false) //need to remove and predict accuracy immediately
            {
                if(outli.classValue() != outli.EPrediction.Predclass)
                {
                    outli.err = true;
                    if(TheNewClass[(int)outli.classValue()] == true)
                    {
                        outli.fn = true;
                        outli.isNovel = true;
                    }
                }
                if(outli.Comitted == false)
                    totUnComitted ++;
                
                Results.Commit(outli);

                //logger.debug("Commited ="+outli.Id +" error  = " + outli.err);
                outlierList.delete(i);
                i --;
            }

            if(!needfilter) //if this is not needed to filter out, then later instances also do not need
                break;
        }//for i (all outliers)
        //logger.debug("Total filtered = " + totUnComitted);
        return totUnComitted; 
    }


    boolean isInside(ArrayList clusters, Instance instance)
    {        
        for(int i = 0; i < clusters.size(); i ++)
        {
            double d = Kmeans.Euclidean(instance,((Cluster)clusters.get(i)).centroid,-1);
            if( d <= ((Cluster)clusters.get(i)).meand) //within mean distance
                return true;
        }
        return false;
    }
    
    public double calculateEnsembleConf(ArrayList<MapPrediction> predictions, int votedClass, int es)
    {
        /* confidence of the ensemble = average confidence of classifiers which have same prediction as
        the final prediction.
        */
        double ensembleConf = 0.0;
        boolean allPredictedOutlier = true;
        /* if all the classifiers have predicted as an outlier, make conf zero; 
         * otherwise, take the average vote towards the predicted class
         */
        for(int i=0; i<predictions.size(); i++)
        {
            if(!predictions.get(i).Isoutlier)
                allPredictedOutlier = false;
            if(Math.abs(predictions.get(i).Predclass - votedClass) < 0.0001)
            {
                ensembleConf += predictions.get(i).weight;
            }
        }
        if(allPredictedOutlier == true)
            return 0.0;
        else
            return ensembleConf/(double)es;
    }

    //test using MineClasS ensemble
    boolean testEnsemble(Classifier[] en, int es, Minstance minst, 
            Minstances outlierList, //outlierList, waiting to be classified
            ArrayList novelClusters, //novel class clusters
            boolean newEnsemble )   //ensemble updated?
            throws Exception
    {          
        int votedClass = 0; //the voted class
        boolean isNewClass = false; //has any new class appeared?

        if(ClassAppeared[(int)minst.classValue()] == false)
        {       
            TheNewClass[(int)minst.classValue()] = true;
        }       

        votedClass = classifySingle(en, es, minst);
        double ensembleConf = calculateEnsembleConf(minst.Predictions, votedClass, es);
        /* add the ensemble confidence and predicted class as another prediction
        all the other fields contain dummy values
        */
        minst.addPrediction(-1, votedClass, false, -1.0 , -1.0,/* -1.0,*/ -1.0, ensembleConf);
        //if(votedClass != minst.classValue())
        //{
            if(!minst.EPrediction.Isoutlier) //classify immediately
            {
                //General error
                //logger.debug("Classify normally....");
                if(votedClass != minst.classValue())
                {
                    minst.err = true;
                    if(TheNewClass[(int)minst.classValue()] == true) //false negative, novel class instance
                    {
                        minst.fn = true;
                        minst.isNovel = true;
                    }
                }
                Results.Commit(minst);
                //logger.debug("Commited ="+minst.Id +" error  = " + minst.err);
            }
            else //if not foutlier
            {
                //see if novel clusters are existing and it falls into them
                boolean committed = false;
                if(novelClusters != null)
                {
                    boolean inside = isInside(novelClusters,minst);
                    if(inside) //then classify as novel class
                    {
                        //logger.debug("Classified as novel class since inside the clusters");
                        //true positive
                        if(TheNewClass[(int)minst.classValue()] == true) 
                        {
                            minst.isNovel = true;
                        }
                        else //false positive
                        {
                            minst.fp = true;
                            minst.err = true;
                        }
                        Results.Commit(minst);
                        committed = true;
                    }//inside                    
                }//ifnovelClusters
                if(!committed)
                    outlierList.add(minst);//process later (put into outlierList)
            }//else outlier
        //}//voted class not correct
        //else
        //{
        //    Results.Commit(minst);//correct prediction
        //}
                       

        //now find whether the outliers are new class =====>        
        //Is it time to check for the new class?

        //1. Remove oldest (if labeled)
        //2. Filter out using current ensemble
        //3. Wait and see

        //REMOVE OLDEST
        if(outlierList.numInstances() == 0)
            return false;


        //check outlierList to see if any one needs to be classified
        int id = 0;
        for(int i = 0; i < outlierList.numInstances(); i ++)
        {
            if(! outlierList.minstance(i).Comitted)
            {
                id = i;
                break; //reached to the oldest uncommited one
            }
            if( outlierList.minstance(i).Id + this.Tl <= minst.Id) //then this instance has been labeled
            {
                if(ClassAppeared[(int)outlierList.minstance(i).classValue()] == true) //then this is not a novel class instance
                {
                    outlierList.delete(i);
                    i --;
                }//if
            }//if
        }//for

        //System.out.println("d = " + id + " total instances = " + outlierList.numInstances());
        
        if(outlierList.numInstances() == 0)
            return false;

        
        int totalUnComitted = outlierList.numInstances() - id;
        
        if(outlierList.minstance(id).Id + this.Tc <= minst.Id) //it must be classified now
        {

            //logger.debug("Deadline to classify the instance approached...");
            Minstance outli = outlierList.minstance(id);

            //update errors
            if(outli.EPrediction.Predclass == outli.classValue())
               ; //then correct
            else
            {
                outli.err = true;                
                if(TheNewClass[(int)outli.classValue()] == true) //false negative
                {
                    outli.fn = true;              //could not detect
                    outli.isNovel = true;                    
                }                
            }
            
            Results.Commit(outli); //classified
            //logger.debug("Commited ="+outli.Id +" error  = " + outli.err);
            //logger.debug("Instance ID=" + minst.Id);
            totalUnComitted --;
        }

        //FILTER

        //First remove outliers that are too old

        if(outlierList.minstance(0).Id + this.ChunkSize <= minst.Id) //remove if older than chunksize
        {
            if(outlierList.minstance(0).Comitted == false) //DOUBT?
            {
                totalUnComitted --;
                System.out.println("Error -- Age too old not to be comitted");
            }
            //logger.debug("Removing the oldest instance from buffer..");
            outlierList.delete(0);
        }
        

        //Now re-evaluate outliers using the current ensemble

        if(newEnsemble)
        {
            int tot = filterOutliers(en, es, outlierList, minst.Id); //how many committed in filtering?
            totalUnComitted -= tot;
            
            newEnsemble = false;
        }

        //double avgsc = 0;
        int newcls = 0; //how many says new cls?                
        Kmeans outkm = null;

        //if there are enough foutliers now
        if(totalUnComitted > this.Minpts && (minst.Id > this.Minpts + LastCheck)) // then check for novel classes
        {

            logger.debug("Lastcheck = " + LastCheck + " Total outliers = " + outlierList.numInstances()
                    + " Current time = " + minst.Id + " Minpts = " + this.Minpts + " Oldest outlier = " + outlierList.minstance(0).Id);

            LastCheck = minst.Id;

            //boolean[] tmpnewcls = new boolean[minst.numClasses()];
            //logger.debug("Checking for novel class.....");
            //first set the isNewClas field
            for(int i = 0; i < outlierList.numInstances(); i ++)
            {
                if(TheNewClass[(int)outlierList.minstance(i).classValue()])
                    //tmpnewcls[(int)outlierList.minstance(i).classValue()] =
                    isNewClass = true;
                outlierList.minstance(i).EPrediction.NumIsNovel = 0;
            }
            
            //compute #of outlier clusters
            double lambda = this.Minpts / 2; //(double)this.ChunkSize / (double)this.K;
            int L = (int)((double)outlierList.numInstances() / lambda);

            //create L outlier clusters
            outkm = new Kmeans(outlierList,L);
            
            if(L > 1)
            {
                logger.debug("Making outlier clusters, total outliers = " + outlierList.numInstances()+" total clusters: "+L);
                outkm.mkOutlierCluster();                          

                int totnc = 0;
                totnc = novelClusters.size();

                Cluster[] allClusters = new Cluster[outkm.Clusters.length + totnc];
                for(int i = 0; i < outkm.Clusters.length; i ++)
                    allClusters[i] = outkm.Clusters[i];

                //append the novel clusters at the end
                if(novelClusters != null)
                {
                    for(int i = outkm.Clusters.length; i < allClusters.length; i ++)
                        allClusters[i] = (Cluster)novelClusters.get(i - outkm.Clusters.length);
                }

                int[] novelVote = new int[allClusters.length];


                //for each classifier, check if it predicted a new class
                for(int i = 0; i < es; i ++)
                {

                    //Get all the clusters for the current model
                    Cluster[] clusters = (Cluster[])en[i].getClass().getDeclaredField("mcClusters").get(en[i]);

                    Outlier o = new Outlier();
                    //double[] sc = new double[data.numClasses() + 1];

                    logger.debug("Classifier ID: "+en[i].getClass().getDeclaredField("Cid").getInt(en[i]));
                    boolean isnc = o.getNSCseparate(clusters, allClusters, Minpts);

                    for(int j = 0; j < allClusters.length; j  ++)
                        if(allClusters[j].sc > 0)
                            novelVote[j] ++;

                    //now unmark the outliers that have s.c < threshold
                    for(int j = 0; j < outlierList.numInstances(); j ++)
                    {
                        Minstance p = outlierList.minstance(j);
                        
                        if(p.ClusterId >= 0 && allClusters[p.ClusterId].sc > 0.0)
                        {
                            p.EPrediction.NumIsNovel ++;
                        }                      
                    }
                    if(isnc)
                            newcls ++;

              
                }//for i

                //If majority predicted a new class........
                if(newcls == es)
                {
                    //System.out.println("New class found");

                    novelClusters = new ArrayList();

                    //first, set up the novelClusters list
                    for(int i = 0; i < allClusters.length; i ++)
                    {
                        if(novelVote[i] == es)
                        {
                            novelClusters.add(allClusters[i]);
                        }
                    }

                    //re-compute errors
                    //because if new class appeared, then some decisions taken earlier were wrong
                    for(int i = 0; i < outlierList.numInstances();  i ++)
                    {

                        Minstance p = outlierList.minstance(i);
                        if(p.Comitted)
                            continue; //already processed

                        int out = p.EPrediction.NumIsNovel;

                        //find if this still remains after filtering
                        if(out == es)
                        {
                            //logger.debug("detected as a novel class instance...");
                            if(isNewClass) //new class really appeared
                            {
                                if(TheNewClass[(int)p.classValue()]) //this is a novel class instance
                                {
                                    p.isNovel = true;
                                }
                                else
                                { // our prediction is wrong (this instance does not belong to a new class)
                                    p.err = true;
                                    p.fp = true; //false positive (because this is F-outlier, but not belongs to a new class)
                                }
                            }
                            else //new class did not appear actually. So, this must be false positive
                            {
                                p.err = true;
                                p.fp = true;
                            }
                            Results.Commit(p);
                            outlierList.delete(i);
                            i --;
                        }
                        else //this was not found to be a novel class instance
                        {
                            //wait a little bit more //if isNewClass
                        }//else this was not found as a novel class

                        //logger.debug("Commited ="+p.Id +" error  = " + p.err);
                    }//for i (all outliers
                    String debug = "NEW CLASS FOUND";
                    if(isNewClass == false)
                    {
                         debug += "  IN ERROR";
                         //System.out.println("Error");
                    }
                    logger.debug(debug);
                    //outlierList.delete(); //no need to keep 'em waiting

                }//if avgsc > 0
                else //new class not detected
                {
                    String debug = "OUTLIERS FOUND ";
                    logger.debug(debug);
                }
            }
        }
        return isNewClass;
        
    }//test ensemble
  
    int updateEnsemble(Classifier[] en, Classifier C, Minstances data) throws Exception
    {

        //String debug = "Update ensemble. Individual weights: ";
        
        //compute weight
        /*for(int i = 0; i < en.length; i ++)
        {
            test(en[i],data);
            debug += " "+en[i].getClass().getDeclaredField("Cid").getInt(en[i])+":"
                    +en[i].getClass().getDeclaredField("Weight").getDouble(en[i]);
        }

        logger.debug(debug);*/
        
        int mi = 0; 
        //boolean updated = false;
        
        //get minimum weight (weight already computed during testing)
        //String debg = " " + en[0].Cid + " "+ en[0].Weight+ ", ";
        for(int i = 1; i < en.length; i ++)
        {
            double en_i_weight  = en[i].getClass().getDeclaredField("Weight").getDouble(en[i]);
            double en_mi_weight = en[mi].getClass().getDeclaredField("Weight").getDouble(en[mi]);
            int en_i_cid        = en[i].getClass().getDeclaredField("Cid").getInt(en[i]);
            int en_mi_cid       = en[mi].getClass().getDeclaredField("Cid").getInt(en[mi]);
            
            if((en_i_weight < en_mi_weight)
                    || ((en_i_weight == en_mi_weight)
                    && (en_i_cid < en_mi_cid)))
            {                
                mi = i;                  
            }//if
            
            //debg += en[i].Cid + " " + en[i].Weight + ", ";
        }//for

        //logger.debug("Minimum weight found for ID=" + en[mi].getClass().getDeclaredField("Cid").getInt(en[mi]));
        
        double c_weight  = test(C,data);
        C.getClass().getDeclaredField("Weight").setDouble(C,c_weight);
        double en_mi_weight = en[mi].getClass().getDeclaredField("Weight").getDouble(en[mi]);

        //logger.debug("New classifier weight= " + c_weight);
        //C.Weight = test(C,data);
        //debg += " "+C.Cid + " " + C.Weight;
        
        //logger.debug("Weights = " + debg);
        
        //if(en_mi_weight <= c_weight) //new classifier is better
        {
            //logger.debug("Worst classifier is replaced");
            en[mi] = C;
            //updated = true;
        }
        
        logger.debug("Filtering ensemble...");
        //remove negative weight classifiers
        int es = en.length;
        for(int i = 0; i < es; i ++)
        {
            double en_i_weight  = en[i].getClass().getDeclaredField("Weight").getDouble(en[i]);
            if(en_i_weight < 0)
            {
                //logger.debug("Deleting "+ en[i].getClass().getDeclaredField("Cid").getInt(en[i]));
                en[i] = en[es - 1]; //overwrite with the last one             
                es --; //one less remaining
                i--;    
            }                
        }
        
        return es;
    }//updateEnsemble
    
    //filter out negative weight classifiers
    int filterEnsemble(Classifier[] en, int es, int chunkNo, Minstances data) throws Exception
    {
        /*String debug = "Filter ensemble. Individual weights: ";
        //compute weight
        for(int i = 0; i < es; i ++)
        {
            test(en[i],data);
            debug += " "+en[i].getClass().getDeclaredField("Cid").getInt(en[i])+":"
                    +en[i].getClass().getDeclaredField("Weight").getDouble(en[i]);
        }

        logger.debug(debug);*/
        
        for(int i = 0; i < es; i ++)
        {
            double en_i_weight  = en[i].getClass().getDeclaredField("Weight").getDouble(en[i]);
            if(en_i_weight < 0)
            {
                //logger.debug("Deleting "+ en[i].getClass().getDeclaredField("Cid").getInt(en[i]));
                en[i] = en[es - 1];
                es --;
                i --;
            }//if
        }//for
        
        //if there are more than one models now
        if(es > 1 && chunkNo >= this.M)
        {
            //find the oldest one
            int old = 0;
            for(int i = 1; i < es; i ++ )
            {
                int en_i_cid        = en[i].getClass().getDeclaredField("Cid").getInt(en[i]);
                int en_old_cid      = en[old].getClass().getDeclaredField("Cid").getInt(en[old]);

                if(en_i_cid < en_old_cid)
                    old = i;
            }
            
            //Check if the oldest model has a class that no other model has
            //boolean[] en_j_Dataseen = ()

            for(int i = 0; i < ((boolean[])en[old].getClass().getDeclaredField("Dataseen").get(en[old])).length; i ++)
            {
                if(((boolean[])en[old].getClass().getDeclaredField("Dataseen").get(en[old]))[i] == true) //check others
                {
                    boolean found = false;
                    for(int j = 0; j < es; j ++)
                    {
                        if(j == old)
                            continue;
                        if(((boolean[])en[j].getClass().getDeclaredField("Dataseen").get(en[j]))[i] == true)
                        {
                            found = true;
                                break;
                        }                        
                    }//for rest of the models
                    if( !found ) //then we must discard old
                    {
                        //logger.debug("Removing the oldest classifier, ID="+en[old].getClass().getDeclaredField("Cid").getInt(en[old]));
                        en[old] = en[es - 1];
                        es --;
                        break;
                    }
                }//if
            }//for each class
        }//if (es > 1)
        return es;
    }

    //Print temporary result

    void printResult(int chunkSize, int K, int Minpts)
    {
        int chunkno = 0;
        this.GlobalError = this.GlobalFN = this.GlobalFP = this.GlobalNC = 0;
        int total = 0;
        if(Results.AllResult.size() == 0)
            return;
        for(chunkno = 0; chunkno < Results.AllResult.size(); chunkno ++)
        {
            ResultStat st = (ResultStat)Results.AllResult.get(chunkno);
            this.GlobalError += st.err;
            this.GlobalFP += st.fp;
            this.GlobalFN += st.fn;
            this.GlobalNC += st.nc;
            total += st.total;

            if(!((ResultStat)Results.AllResult.get(chunkno)).printed)
                break;            
        }
        ResultStat st = (ResultStat)Results.AllResult.get(chunkno);
        //System.out.println("before checking st.full, which is ="+st.full()+ " chunk no = " + chunkno);
        if(!st.full())
            return;

        try{
            //System.out.println("Here--inside try");
            BufferedWriter out = null;
            if(chunkno == 0) //first chunk
            {                
                 out = new BufferedWriter(new FileWriter("SAND_"+ "V" + Constants.VERSION +"-"+(InFile.lastIndexOf("/")==-1?InFile.substring(InFile.lastIndexOf("\\")+1):InFile.substring(InFile.lastIndexOf("/")+1))+"-T"+this.Tl+
                 "-C"+this.Tc+ "-S"+this.ChunkSize+"-U"+Constants.TAU+"-D"+Constants.DYNAMIC + ".tmpres"));
                 out.write("Chunk#\tFP\tFN\tNC\tErr\tGlobErr\n" );
            }
            //else
            {
                 out = new BufferedWriter(new FileWriter("SAND_"+ "V" + Constants.VERSION +"-"+(InFile.lastIndexOf("/")==-1?InFile.substring(InFile.lastIndexOf("\\")+1):InFile.substring(InFile.lastIndexOf("/")+1))+"-T"+this.Tl+
                 "-C"+this.Tc+ "-S"+this.ChunkSize+"-U"+Constants.TAU+"-D"+Constants.DYNAMIC+".tmpres",true));
                String output = chunkno+"\t"+st.fp+"\t"+
                        st.fn+"\t"+
                        st.nc+"\t"+
                        st.err+"\t"+
                        (double)this.GlobalError * 100.0/(double)total+"\n";

                out.write(output);
                //System.out.print(output);
                st.printed = true;
            }
            if(out != null)
                out.close();
        }
        catch(IOException e)
        {
            System.out.println(e.getMessage());
        }
    }
   
    //Print summary result
    void printResult(String finit, int totInstance,int K, int M)
    {
        try{
            BufferedWriter    out = new BufferedWriter(new FileWriter("SAND_"+ "V" + Constants.VERSION +"-"+(InFile.lastIndexOf("/")==-1?InFile.substring(InFile.lastIndexOf("\\")+1):InFile.substring(InFile.lastIndexOf("/")+1))+"-T"+this.Tl+
                 "-C"+this.Tc+ "-S"+this.ChunkSize+"-U"+Constants.TAU+"-D"+Constants.DYNAMIC+".res"));
            out.write("Chunk#\tFP\tFN\tNC\tErr\n" );

            double fp = 100 * (double)GlobalFP / (double)(totInstance - GlobalNC);
            double fn = 100 * (double)GlobalFN / (double)(GlobalNC);
            double nc = GlobalNC;
            double err = 100 * (double) GlobalError / (double)totInstance;
            
            
            out.write("\nFP%\tFN%\tNC(total)\tErr%\n");
            out.write(fp +"\t"+fn+"\t"+GlobalNC+"\t"+err) ;
            
            out.write("\nMethod\tTraintime per 1K (ms):\tTesttime per 1K (ms)\tTottime per 1K (ms)\n");
            out.write("MineClasS\t"+1000.0*TrainTime/(double)totInstance+"\t"+1000.0 * TestTime/(double)totInstance+"\t"+1000.0*(TrainTime+TestTime)/(double)totInstance+"\n");
            out.write("Processing speed:\t"+1000.0*(double)totInstance/(double)(TrainTime+TestTime)+"\n");
            out.close();
            
        }
        catch(IOException e)
        {
            System.out.println(e.getMessage());
        }     
    }
    
    long addUpdateTime()
    {
        long temp = System.currentTimeMillis();
        long ret = temp - CurTime;
        CurTime = temp;
        return ret;
    }
           
    void Classify(String[] argv)
    {
        CurTime = System.currentTimeMillis();        
        String infile = this.InFile;               //base file name                 
        String classifierName = this.ClassifierName;      
        Classifier[] En = new Classifier[M];

        int numOfLabeledInst = 0;
        int numOfUnlabeledInst = 0;
        Random rand = new Random();
       
        // for debugging purpose only, disable after debugging
        /*
        BufferedWriter changePointWindowVals = null;
        BufferedWriter windowSizeVsError = null;
        try
        {
            changePointWindowVals = new BufferedWriter(new FileWriter("window_vals"));
            windowSizeVsError = new BufferedWriter(new FileWriter("window_size_vs_error.csv"));
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        */
        // for debugging purpose only, disable after debugging
        
        int es = 0; //ensemble size
        int cid = 0; //classifier id
        //T = 1000; //time delay for labeling
        int numdata = 0; //how many data has been tested?

        BufferedReader in = null;
        Instances data = null;

        //Build initial ensemble
        try{
            in = new BufferedReader(new FileReader(infile+".arff"));
            data = new Instances(in,this.ChunkSize);
            data.setClassIndex(data.numAttributes()-1);
            Allinst = new BufferedWriter(new FileWriter("Allinst.out"));
            
            for(int i = 0; i < Constants.WARMUP; i ++)
            {
                while(data.readInstance(in))
                {
                    //System.out.println(data.numInstances() + " " + ((Instance)data.instance(data.numInstances() - 1)).value(0));
                    if(data.numInstances() == this.ChunkSize)
                    break;
                }
                
                //build initial classifier                
                Classifier C = Classifier.forName(classifierName, argv);
                C.buildClassifier(data);
                //C.getClass().getDeclaredField("NumData").setDouble(C, data.numInstances());
                double c_weight  = testScratch(C,data);
                //C.getClass().getDeclaredField("TrainWeight").setDouble(C,c_weight);
                C.getClass().getDeclaredField("Weight").setDouble(C,c_weight);
                C.getClass().getDeclaredField("Cid").set(C, cid++);
                En[i] = C;
                es ++;
                if(i != Constants.WARMUP - 1)
                {
                    data.delete();
                }
            }//for i            
        }//try
        catch(Exception e)
        {
            System.out.println(e.getMessage());
        }
        
        /* find the coeff vals based on the latest chunk of data */
        /* I consider two metrics
        1. distance from centroid
        2. Ratio of data instances in that cluster supporting the max class
        */
        double[][] distFromPer = new double[es][data.numInstances()];
        double[][] ratioMaxClass = new double[es][data.numInstances()];
        double[][] isCorrectClassification = new double[es][data.numInstances()];
        
        for(int i=0; i<data.numInstances(); i++)
        {
            Minstance minst = new Minstance(data.instance(i), -1);
            //First test the instance
            int votedClass = -1;
            try
            {
                votedClass = this.classifySingle(En, es, minst);    
            }
            catch(Exception e)
            {
                e.printStackTrace();
            }
            for(int j=0; j<minst.Predictions.size(); j++)
            {                
                MapPrediction mapPrediction = (MapPrediction)minst.Predictions.get(j);
                distFromPer[j][i] = mapPrediction.distFromPer;
                ratioMaxClass[j][i] = mapPrediction.ratioMaxClass;
                if(mapPrediction.Predclass == minst.classValue())
                {
                    isCorrectClassification[j][i] = 1;
                }
                else
                {
                    isCorrectClassification[j][i] = 0; 
                }    
            }
        }
        /*now find the correlation */
        double[][] correlationVals = new double[es][2];
        for(int i=0; i<es; i++)
        {
            correlationVals[i][0] = findCorrelation(distFromPer[i], isCorrectClassification[i]);
            correlationVals[i][1] = findCorrelation(ratioMaxClass[i], isCorrectClassification[i]);
        }

        for(int i=0; i<correlationVals.length; i++)
        {
            double sum = 0.0;
            for(int j=0; j<correlationVals[i].length; j++)
            {
                sum += correlationVals[i][j];
            }
            if(sum == 0.0)
                sum = 0.00001;
            for(int j=0; j<correlationVals[i].length; j++)
            {
                correlationVals[i][j] /= sum;
            }
        }
        /* finally set the values in the classifiers */
        /* here add the accuracy of each classifier on last training data chunk */
        for(int i=0; i<es; i++)
        {
            try
            {
                ConfidenceStats confStats = (ConfidenceStats)En[i].getClass().getDeclaredField("confidenceStats").get(En[i]);
                confStats.clearContents(); //to make sure that it does not contain anything from the history.
                /* set min max */
                confStats.getMinConfVals().add(findMinVal(distFromPer[i]));
                confStats.getMinConfVals().add(findMinVal(ratioMaxClass[i]));
                
                confStats.getMaxConfVals().add(findMaxVal(distFromPer[i]));                               
                confStats.getMaxConfVals().add(findMaxVal(ratioMaxClass[i]));                
                /* set the correlation coefficient */
                for(int j=0; j<2; j++)
                {
                    confStats.getCorrCoeff().add(correlationVals[i][j]); 
                }
                confStats.accuracy = findAccuracy(isCorrectClassification[i]);
            }
            catch(Exception e)
            {
                e.printStackTrace();
            }
        }
        /* done finding coeff vals */
        data.delete();
        
        ClassAppeared = new boolean[data.numClasses()];
        TheNewClass = new boolean[data.numClasses()];

        //initialize class appeared flag
        for(int i = 0; i < data.numClasses(); i ++)
        {
            ClassAppeared[i] = false;
            TheNewClass[i] = false;
            for(int j = 0; j < es; j ++)
            {
                try{
                    ClassAppeared[i] = ClassAppeared[i] |
                        ((boolean[])En[j].getClass().getDeclaredField("Dataseen").get(En[j]))[i];
                }
                catch(Exception e)
                {
                    System.out.println(e.getMessage());
                }
            }//for j
        }//for i

        //sort into reverse order (most recent first)
        for(int i = 0; i < es - i; i ++)
        {
            Classifier t = En[i];
            En[i] = En[es - i - 1];
            En[es - i - 1] = t;
        }
        Minstances trainData = new Minstances(data, 0);
        Minstances testData = new Minstances(data, 0);
        Minstances outlierList = new Minstances(data, 0);
        ArrayList novelClusters = new ArrayList();

        int IId = 0; //instance id (unique)
        int ensno = 0; //ensemble number

        boolean newEnsemble = false;
        /* debug-start */
        /*
        BufferedWriter confWriter = null;
        try
        {
            File confValFile = new File("conf_vals.txt");
            confWriter = new BufferedWriter(new FileWriter(confValFile)); 
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        */
        /* debug-end */
        
        /* declare the change detection object */
        BetaDistributionChangePoint changeDetector = new BetaDistributionChangePoint((1/2), 0.001);
        //to monitor the error in the current window, since we dont save the error in the dynamic window for SAND
        ArrayList<Double> errorMonitor = new ArrayList<Double>(); 
        int estimatedCP = -1;
        
        try{
             while(data.readInstance(in))
             {
                //classify
                Instance inst = data.instance(0);
                data.delete(0);
                //make a Minstance
                Minstance minst = new Minstance(inst, IId++);
                testData.add(minst);
                //First test the instance
                this.testEnsemble(En, es, minst, outlierList, novelClusters, newEnsemble);
                /* debug start */
                /*
                if(minst.Id == 8500)
                {
                    System.out.println("hello");
                }
                */
                /* debug end */
                newEnsemble = false;
                TestTime += addUpdateTime();                
                if(numdata % 500 == 0)
                {
                    this.printResult(1000, K,  Minpts);
                }
                numdata ++;
                /* for debugging only*/
                if(numdata % 100 == 0)
                {
                    System.out.println("Num Data val: " + numdata);
                }
                //System.out.println(numdata);
                //See if any instance in the data buffer has been labeled
                if(testData.numInstances() > Tl)
                {
                    //trainData.add(testData.minstance(0));
                    MapPrediction ensemblePrediction = (MapPrediction)testData.minstance(0).Predictions.get(testData.minstance(0).Predictions.size()-1);
                    if(ensemblePrediction.Cid != -1)
                        ensemblePrediction = extractEnsemblePrediction(testData.minstance(0).Predictions);
                    if(ensemblePrediction != null)
                    {
                        //add in the training data set based on confidence
                        /* if classifier confidence is low, then request for
                        labels from the user.
                        else if the confidence is very high, then no need to
                        acquire the label, rather use the prediction itself as
                        the label.
                        */
                        if(ensemblePrediction.weight <= Constants.TAU)
                        {
                            trainData.add(testData.minstance(0));
                            numOfLabeledInst++; 
                        }
                        else
                        {
                            testData.minstance(0).setClassValue(ensemblePrediction.Predclass);
                            trainData.add(testData.minstance(0));
                            numOfUnlabeledInst++;
                        }
                        //smoothing
                        if(ensemblePrediction.weight < 0.1)
                            changeDetector.insertIntoWindow(0.05);
                        else if(ensemblePrediction.weight > 0.995)
                            changeDetector.insertIntoWindow(0.995);
                        else
                            changeDetector.insertIntoWindow(ensemblePrediction.weight);
                        //System.out.print("IId:" + IId + ":" + ensemblePrediction.weight + ", ");
                    }
                    else
                    {
                        System.err.println("Ensemble prediction is null!!!");
                    }
                    //save the error on the test instance
                    if(testData.minstance(0).err == true)
                    {
                        //error occurred, so push 0 to the change detector
                        //System.out.println("numdata: " + numdata + ";predition is: " + testData.minstance(0).EPrediction.Predclass + "; error");
                        errorMonitor.add(1.0);
                    }
                    else
                    {
                        //no error, so push 1 to the change detector
                        //System.out.println("numdata: " + numdata + ";predition is: " + testData.minstance(0).EPrediction.Predclass + "; no error");
                        errorMonitor.add(0.0);
                    }
                    /* debug_start*/
                    /*
                    try
                    {
                        //confWriter.write("IId:" + IId + ":" + ensemblePrediction.weight + ", ");
                        confWriter.write("Instance index: " + testData.minstance(0).Id + "\tTrue label: " + testData.minstance(0).classValue() + "\tPredicted Label: " + 
                                ensemblePrediction.Predclass + "\tError?: "+ testData.minstance(0).err +"\tConfidence: "+ensemblePrediction.weight + "\n");
                        confWriter.flush();
                    }
                    catch(Exception e)
                    {
                        e.printStackTrace();
                    }
                    */
                    /* debug_end*/
                    //write in windowSizeVsError
                    /*
                    if((IId%3)==0)
                    {
                        windowSizeVsError.write("IID," + IId + ",Window_Size," + changeDetector.getDynamicWindow().size() + ",Error,"+(100 * (double) changeDetector.calculateListMean(errorMonitor, 0, errorMonitor.size()-1))+"\n");
                        windowSizeVsError.flush();
                    }
                    */
                    if(!Constants.DYNAMIC){
                        if(ensemblePrediction.weight <= Constants.TAU)
                        {
                            estimatedCP = changeDetector.detectChange();                        
                        }
                        else
                        {
                            estimatedCP = -1;
                        }
                    }
                    else{
                        if(rand.nextDouble() <= Math.exp(-1 * ensemblePrediction.weight))
                        {
                            estimatedCP = changeDetector.detectChange();    
                        }
                        else
                        {
                            estimatedCP = -1;
                        }
                    }
                    testData.delete(0);
                }
                //time to train
                if(/*trainData.numInstances() == this.ChunkSize*/estimatedCP != -1)
                {
                    /*
                    System.out.println("\nEstimatedCP = " + estimatedCP + "\nIId = "+IId+"\n" + 
                            "\nIndex of last training instance = " + trainData.minstance(trainData.numInstances()-1).Id +
                            "\nIndex of last testing instance = " + testData.minstance(testData.numInstances()-1).Id);
                    // for debugging purpose only, disable after debugging
                    String windowContents = "[";
                    for(int i=0; i<changeDetector.getDynamicWindow().size(); i++)
                    {
                        windowContents += changeDetector.getDynamicWindow().get(i) + ";";
                    }
                    windowContents += "]";
                    
                    changePointWindowVals.write(windowContents + "\n");
                    changePointWindowVals.flush();
                    */
                    // for debugging purpose only, disable after debugging          
                    if(Constants.DYNAMICNUMOFCLUSTERS/* && trainData.numInstances() < 500*/)
                    {
                        argv[0] = "-K";
                        argv[1] = (int)Math.floor(trainData.numInstances()/50) + ""; 
                    } 

                    Classifier C = Classifier.forName(classifierName, argv);
                    C.buildClassifier(trainData);
                    String debg = "Classifier ID = " +cid+ " Existing classes = ";
                    for(int i = 0; i < minst.numClasses(); i ++)
                    {
                        boolean present = ((boolean[])C.getClass().getDeclaredField("Dataseen").get(C))[i];
                        if(present)
                            debg += " "+i;
                    }
                    logger.debug(debg);                    
                    C.getClass().getDeclaredField("Cid").set(C, cid ++);
                    long commTime = addUpdateTime(); //common time
                    TrainTime += commTime;
                    //C.buildClassifier(data);
                    TrainTime += addUpdateTime();  
                    //System.out.println("Chunk no" + i + ", Ensemble size = " + es);
                    if(es == M)  //quota filled up
                    {
                        es = updateEnsemble(En, C, trainData);
                        TrainTime += addUpdateTime();
                        //System.out.println("Ensemble updated");
                    }
                    else
                    {
                        es = filterEnsemble(En,es, cid, trainData);
                        double weight = testScratch(C,trainData);
                        C.getClass().getDeclaredField("Weight").set(C, weight);
                        if(weight > 0 || es == 0)
                        {
                            En[es ++] = C;
                        }
                        TrainTime += addUpdateTime();
                    }//else
                    /* now update the confidence statistics */
                    distFromPer = new double[es][trainData.numInstances()];
                    ratioMaxClass = new double[es][trainData.numInstances()];       
                    isCorrectClassification = new double[es][trainData.numInstances()];                   
                    for(int i=0; i<trainData.numInstances(); i++)
                    {
                        minst = trainData.minstance(i);
                        minst.clearPredictions();
                        minst.Id = -1;
                        //First test the instance
                        int votedClass = -1;
                        try
                        {
                            votedClass = this.classifySingle(En, es, minst);    
                        }
                        catch(Exception e)
                        {
                            e.printStackTrace();
                        }
                        for(int j=0; j<minst.Predictions.size(); j++)
                        {                
                            MapPrediction mapPrediction = (MapPrediction)minst.Predictions.get(j);
                            distFromPer[j][i] = mapPrediction.distFromPer;
                            ratioMaxClass[j][i] = mapPrediction.ratioMaxClass;
                            if(mapPrediction.Predclass == minst.classValue())
                            {
                                isCorrectClassification[j][i] = 1;
                            }
                            else
                            {
                                isCorrectClassification[j][i] = 0; 
                            }   
                        }
                    }
                    /*now find the correlation */
                    correlationVals = new double[es][3];
                    for(int i=0; i<es; i++)
                    {
                        correlationVals[i][0] = findCorrelation(distFromPer[i], isCorrectClassification[i]);
                        correlationVals[i][1] = findCorrelation(ratioMaxClass[i], isCorrectClassification[i]);
                    }
                    /* need to scale the correlation values so that sum will be 1.0*/
                    /* issue: sometimes while rescaling, correlation has value more than 1.0, 
                     * specially when some original vals are negative and some vals are positive.
                     */
                    for(int i=0; i<correlationVals.length; i++)
                    {
                        double sum = 0.0;
                        for(int j=0; j<correlationVals[i].length; j++)
                        {
                            sum += correlationVals[i][j];
                        }
                        if(sum == 0.0)
                            sum = 0.00001;
                        for(int j=0; j<correlationVals[i].length; j++)
                        {
                            correlationVals[i][j] /= sum;
                        }
                    }

                    /* finally set the values in the classifiers */
                    /* here add the accuracy of each classifier on last trainind data chunk */
                    for(int i=0; i<es; i++)
                    {
                        try
                        {
                            ConfidenceStats confStats = (ConfidenceStats)En[i].getClass().getDeclaredField("confidenceStats").get(En[i]);
                            confStats.clearContents(); //to make sure that it does not contain anything from the history.
                            /* set min max */
                            confStats.getMinConfVals().add(findMinVal(distFromPer[i]));
                            confStats.getMinConfVals().add(findMinVal(ratioMaxClass[i]));

                            confStats.getMaxConfVals().add(findMaxVal(distFromPer[i]));                               
                            confStats.getMaxConfVals().add(findMaxVal(ratioMaxClass[i]));                
                            /* set the correlation coefficient */
                            for(int j=0; j<2; j++)
                            {
                                confStats.getCorrCoeff().add(correlationVals[i][j]); 
                            }
                            confStats.accuracy = findAccuracy(isCorrectClassification[i]);
                        }
                        catch(Exception e)
                        {
                            e.printStackTrace();
                        }
                    }
                    /* end updating the confidence statistics */

                    for(int i = 0; i <= estimatedCP; i++)
                    {
                        trainData.delete(0);
                    }
                    /* now delete from change Detector window also */
                    changeDetector.shrinkWindow(estimatedCP);
                    //now delete from errorMonitor also
                    changeDetector.shrinkList(errorMonitor, estimatedCP);
                    //sort (bubble) the classifiers acording to their id (most recent first)
                    for(int i = 0; i < es - 1; i ++)
                        for(int j = i + 1; j < es; j ++)
                        {
                            if(En[i].getClass().getDeclaredField("Cid").getInt(En[i]) <
                                    En[j].getClass().getDeclaredField("Cid").getInt(En[j])) //swap
                            {
                                Classifier t = En[i];
                                En[i] = En[j];
                                En[j] = t;
                            }
                        }
                    //initialize class appeared flag
                    for(int i = 0; i < minst.numClasses(); i ++)
                    {
                        ClassAppeared[i] = false;
                        TheNewClass[i] = false;
                        for(int j = 0; j < es; j ++)
                            ClassAppeared[i] = ClassAppeared[i] |
                                    ((boolean[])En[j].getClass().getDeclaredField("Dataseen").get(En[j]))[i];

                    }
                    newEnsemble = true;
                    novelClusters = new ArrayList();
                    ensno ++;
                }//time to train
             }//while
             System.out.println("\n\n% of labeled data = " + ((double)numOfLabeledInst/(double)(numOfLabeledInst + numOfUnlabeledInst)));
             in.close();
             Allinst.close();
            }//try
            catch(Exception e)
            {
                System.out.println(e.getMessage());
                e.printStackTrace();
            }//catch
                   
        printResult(infile.substring(0,4),IId, K, M);
    }//classify
    
    public MapPrediction extractEnsemblePrediction(ArrayList<MapPrediction> predictions)
    {
        MapPrediction ensemblePrediction = null;
        for(int i=0; i<predictions.size(); i++)
        {
            if(predictions.get(i).Cid == -1)
            {
                ensemblePrediction = predictions.get(i);
            }
        }
        return ensemblePrediction;
    }
    
    public double findCorrelation(double[] confVals, double[] isCorrectVals)
    {
        double correlation = -2.0;
        double meanConfVals = calcMean(confVals);
        double stdConfVals = Math.sqrt(calcVar(confVals));
        double meanIsCorrectVals = calcMean(isCorrectVals);
        double stdIsCorrectVals = Math.sqrt(calcVar(isCorrectVals));
        /* find the covariance value */
        double covariance = 0.0;
        for(int i=0; i<confVals.length; i++)
        {
            covariance += (confVals[i]-meanConfVals) * (isCorrectVals[i]-meanIsCorrectVals);
        }
        covariance /= (confVals.length);
        
        /* finally find the correlation */
        if(stdConfVals == 0.0)
            stdConfVals = 0.00001;
        if(stdIsCorrectVals == 0.0)
            stdIsCorrectVals = 0.00001;
        correlation = covariance/(stdConfVals * stdIsCorrectVals);
        return correlation;
    }

    public double findAccuracy(double[] data)
    {
        int sum = 0;
        for(int i=0; i<data.length; i++)
        {
            if(data[i] == 1.0)
                sum++;
        }
        return (double)sum/(double)data.length;
    }
    
    public double calcMean(double[] data)
    {
    	double sum = 0.0;
    	for(int i=0; i<data.length; i++)
    	{
    		sum += data[i];
    	}
    	return sum/data.length; 
    }
    
    public double calcVar(double[] data)
    {
        double sumOfSquares = 0.0;
        double mean = calcMean(data);
        for(int i=0; i<data.length; i++)
        {
            sumOfSquares += (data[i] - mean) * (data[i] - mean);
        }
        return sumOfSquares/(data.length-1); 
    }
    
    public void normalize(double[] data)
    {
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for(int i=0; i<data.length; i++)
        {
            if(data[i] > max)
                max = data[i];
            if(data[i] < min)
                min = data[i];
        }
        //now normalize ith row of the array
        for(int i=0; i<data.length; i++)
        {
            data[i] = (data[i]-min)/(max-min);
        }
    }
    
    public double findMinVal(double[] data)
    {
        double min = Double.MAX_VALUE;
        for(int i=0; i<data.length; i++)
        {
            if(min > data[i])
                min = data[i];
        }
        return min;
    }
    
    public double findMaxVal(double[] data)
    {
        double max = Double.MIN_VALUE;
        for(int i=0; i<data.length; i++)
        {
            if(max < data[i])
                max = data[i];
        }
        return max;
    }
     /*OPTION HANDLER METHODS*/
   /*
   * Returns an enumeration describing the available options.
   *  
   * @return an enumeration of all the available options.
   */

    @Override
  public Enumeration listOptions() {

    Vector newVector = new Vector(9);

    newVector.
	addElement(new Option("\tShow help",
			      "h", 0, "-h"));
    newVector.
	addElement(new Option("\tSet the base file name (w/o arff extension).",
			      "F", 1, "-F <base file name>"));
    newVector.
	addElement(new Option("\tSet Warm-up Period chunk size" +
			      "\t(default 2000)",
			      "S", 1, "-S <chunk size>"));
    newVector.
	addElement(new Option("\tSet the ensemble size." +
			      "\t(default 6)",
			      "L", 1, "-L <ensemble size>"));
    newVector.
	addElement(new Option("\tSet the confidence threshold." +
			      "\t(default 0.90)",
			      "U", 1, "-U <confidence threshold>"));
    newVector.
	addElement(new Option("\tSAND-D (use 1)/SAND-F (use 0)?" +
			      "\t(default 1 (SAND-D)",
			      "D", 1, "-D <1/0>"));
    newVector.
	addElement(new Option("\tSet time delay for labeling." +
			      "\t(default 1)",
			      "T", 1, "-T <time delay>"));
    newVector.
	addElement(new Option("\tSet time constraint for classification." +
			      "\t(default 0)",
			      "C", 1, "-C <classification delay>"));
  
    /*
    newVector.
	addElement(new Option("\tSet the total number clusters." +
			      "\t(default 50)",
			      "N", 1, "-N <number of clusters>"));

    newVector.
	addElement(new Option("\tSet Minpts." +
			      "\t(default 50)",
			      "M", 1, "-M <Minpts>"));

    newVector.
	addElement(new Option("\tSet the classifier name.",
			      "B", 1, "-B <classifier name>"));
    */
    
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

    this.ShowHelp = Utils.getFlag('h', options);
    
    String inFile = Utils.getOption('F', options);
    if (inFile.length() != 0) {
      this.InFile = inFile;
    } 

    String tdString = Utils.getOption('T', options);
    if (tdString.length() != 0) {
      this.Tl = Integer.parseInt(tdString);
	  }
     else {
      //this.Tl = 1000;
      //default for classification without novel class detection
       this.Tl = 1;
    }

    String tcString = Utils.getOption('C', options);
    if (tcString.length() != 0) {
      this.Tc = Integer.parseInt(tcString);
	  }
     else {
      //this.Tc = this.Tl / 2;
      //default for classification without novel class detection
        this.Tc = 0;
    }
   
    
    String csString = Utils.getOption('S', options);
    if (csString.length() != 0) {
      this.ChunkSize = Integer.parseInt(csString);
	  }
     else {
      //this.ChunkSize = 1000;
        this.ChunkSize = 2000;
    }

    String numClusters = Utils.getOption('N', options);
    if (numClusters.length() != 0) {
        this.K = Integer.parseInt(numClusters);
      }
     else {
      this.K = 50;
    }

    //ensemble size
    String ensString = Utils.getOption('L', options);
    if (ensString.length() != 0) {
        this.M = Integer.parseInt(ensString);
      }
     else {
      this.M = 6;
    }

    //min points
    String minptsString = Utils.getOption('M', options);
    if (minptsString.length() != 0) {
        this.Minpts = Integer.parseInt(minptsString);
      }
     else {
      this.Minpts = 50;
    }

    //min points
    String clsName = Utils.getOption('B', options);
    if (clsName.length() != 0) {
        this.ClassifierName = clsName;
      }
     else {
      this.ClassifierName = "reasc.ReascCtrl";
    }
    
    String tau = Utils.getOption('U', options);
    if (tau.length() != 0) {
        Constants.TAU = Double.parseDouble(tau);
      }
     else {
      Constants.TAU = 0.90;
    }
    
    String isDyn = Utils.getOption('D', options);
    if (isDyn.length() != 0) {
        Constants.DYNAMIC = Integer.parseInt(isDyn)==1;
      }
     else {
      Constants.DYNAMIC = true;
    }
   // this.K = (int) (2.0 * (double)this.ChunkSize / (double)this.Minpts);
    //Constants.K = this.K;
  } //setOptions

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
    @Override
  public String [] getOptions() {

    String [] options = new String [8];
    int current = 0;

    options[current++] = "-F"; options[current++] = "" + this.InFile;
    options[current++] = "-S"; options[current++] = "" + this.ChunkSize;
    options[current++] = "-T"; options[current++] = "" + this.Tl;
    options[current++] = "-C"; options[current++] = "" + this.Tc;
    options[current++] = "-N"; options[current++] = "" + this.K;
    options[current++] = "-L"; options[current++] = "" + this.M;
    options[current++] = "-M"; options[current++] = "" + this.Minpts;
    options[current++] = "-B"; options[current++] = "" + this.ClassifierName;
    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

    public static void printOptions(Enumeration en1, Enumeration en2)
    {
        while(en1.hasMoreElements())
        {
            Option opt = (Option)en1.nextElement();
            System.out.print(opt.synopsis()+" ");
        }

        System.out.println("\nDescription...........");
        //en = ev.listOptions();
        //System.out.print("Miner ");
        while(en2.hasMoreElements())
        {
            Option opt = (Option)en2.nextElement();
            System.out.println(opt.synopsis() + "\n" + opt.description());
        }

        System.out.println("......................\n");
    }

    public static void main(String[] args)
    {
        Miner ev = new Miner();
        
        try{
                ev.setOptions(args);
        }
        catch(Exception e)
        {
            System.out.println("Exception in options handling " + e.getMessage());
        }
        if(ev.InFile.length() == 0 || ev.ClassifierName.length() == 0 || ev.ShowHelp == true)
        {           
            Enumeration en1 = ev.listOptions(), en2 = ev.listOptions();
            
            System.out.print("Miner ");
            printOptions(en1, en2);
            
            if(ev.ClassifierName.length() > 0)
            {
                System.out.println("Classifier Options:");
                try{
                    Classifier C = Classifier.forName(ev.ClassifierName, null);
                    en1 = C.listOptions();
                    en2 = C.listOptions();
                    System.out.print(ev.ClassifierName + " ");
                    printOptions(en1, en2);
                }
                catch(Exception e)
                {
                    System.out.println("No classifier " + ev.ClassifierName + " exists");
                }
            }//if ev.ClassifierName == 0
            return ;
        }//if ev.InFile.length == 0
        
        FileAppender fa=null;
        SimpleLayout sl = new SimpleLayout();
        try{
         fa= new FileAppender(sl, "SAND_"+"V" + Constants.VERSION +"-"+(ev.InFile.lastIndexOf("/")==-1?ev.InFile.substring(ev.InFile.lastIndexOf("\\")+1):ev.InFile.substring(ev.InFile.lastIndexOf("/")+1))+"-T"+ev.Tl+
                 "-C"+ev.Tc+ "-S"+ev.ChunkSize+"-U"+Constants.TAU+"-D"+Constants.DYNAMIC+".log",false);
        }
        catch(IOException e)
        {
            System.out.println(e.getMessage());
        }
        BasicConfigurator.configure(fa);
       
        logger.setLevel(Level.DEBUG);
        
        ev.Classify(args);    
        /*<basefilename> <totChunks><K><M><Minpts> <Classifier args> 
        /*KDD/arff/kdd   200         50 8 50       weka.classifiers.trees.J48ctrl -U*/
        /*KDD/arff1000/kdd 200 0 50 8 50 weka.classifiers.trees.J48ctrl -U*/
    }
}
