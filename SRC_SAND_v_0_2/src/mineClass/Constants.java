package mineClass;

import java.util.*;

import org.apache.log4j.*;

public class Constants {
   
    public static Logger  logger = Logger.getLogger(Constants.class);
    public static final int NIL = -1;   
    public static String[] ClsLabel = null;  //Labels of different classes   
    public static final int Seed = 1234567; //seed for random number generator
    public static final int WARMUP = 6;     //how many chunks are used for warm up?
    public static final double OUTTH = 0.7; //Outlier threshold (lower weight means outlier)
    public static final boolean DYNAMICNUMOFCLUSTERS = true;
    /* if USECONFTOWARDSVOTING is true, then confidence is used as weight while
     * classifying. Otherwise normal voting is done. In either case, confidence 
     * is used for capturing concept drift.
     */
    public static final boolean USECONFTOWARDSVOTING = true;
    public static final boolean FIXED_THRESHOLD = true;
    public static final double VERSION = 0.2;
    public static double TAU = 0.9;
    public static boolean DYNAMIC = true;
   
    /** Creates a new instance of EvoDT */
    public Constants() {
    }

    public static void setLabel(ArrayList labels)
    {
        ClsLabel = new String[labels.size()];
        for(int i = 0; i < labels.size(); i ++)
            ClsLabel[i] = labels.get(i) +"";
    }

    //get the classId of this label
    public static int getLabelId(String label)
    {
        if(ClsLabel == null)
            return NIL;
        for(int i = 0; i < ClsLabel.length; i ++)
            if(label.equals(ClsLabel[i]))
                return i;
        return NIL;
    }
    
    
}
