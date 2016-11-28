/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mineClass;

public class MapOPrediction{
    public int Mid = -1;               //Model Id
    public boolean IsUnknown = false;  //is unknown by this model?
    public boolean IsNovel = false;    //is novel class by this model?

    public MapOPrediction(int mid, boolean isUnknown)
    {
        Mid = mid;
        IsUnknown = isUnknown;
    }
}
