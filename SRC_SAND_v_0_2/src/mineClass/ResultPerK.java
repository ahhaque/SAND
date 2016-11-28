package mineClass;

import java.util.*;
import java.io.*;

public class ResultPerK{
    
    public ArrayList AllResult = new ArrayList();
    /** Creates a new instance of Result */
    public ResultPerK() {
    }
    
    public void Commit(Minstance inst)
    {

        if(inst.Comitted)
            return;

        int bin = inst.Id / 1000; //which bin to place?
        //String debug = "Instance id = " + inst.Id +" bin = " + bin;
        
        if(AllResult.size() <= bin)
        {
            int cur = AllResult.size();
            for(int i = cur; i <= bin; i ++)
                AllResult.add(new ResultStat());
        }
        ResultStat s = (ResultStat)AllResult.get(bin);

        //debug += "  Allresult size = " + AllResult.size();

        if(s.full())
            System.out.println("Error - trying to insert into full bin");
        else
        {
            s.addStat(inst);
            AllResult.set(bin, s);
            inst.Comitted = true;
        }   
    }   
}
