/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mineClass;

/**
 *
 * @author root
 */
//maps an instance from one dataset to another dataset

public class Mapping {

     public int value = -1;
     public int clusterId = -1;
     public Mapping()
     {
         
     }
     public Mapping(int val, int clId)
     {
         value = val;
         clusterId = clId;
     }
}
