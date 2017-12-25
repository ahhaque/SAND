# SAND
SAND: Semi-Supervised Adaptive Novel Class Detection and Classification over Data Stream

## Synopsis
SAND is a semi-supervised framework for classifying evolving data streams. Unlike many other existing approaches, it detects concept drifts in an unsupervised way by detecting changes in classifier confidences in classifying test instances. It also addresses concept evolution problem by detecting outliers having strong cohesion among themselves. Please refer the paper given below for a details description of the approach. 

## Requirements
SAND requires that
* Input file will be provided in .arff format.
* All the features need to be numeric. If there is any non-numeric featues, it should be converted using standard techniques prior using with SAND.
* Features should be normalized to get better performance. 

## Environment
* Java SDK v1.7+
* Weka 3.6+
* Common Math library v2.2
* Apache Logging Services v1.2.15

All of above except java sdk are included inside SRC_SAND_v_0_1 & DIST_SAND_v_0_1 folders.

## Execution
To execute the program, use the following steps:
1. Open a command prompt inside DIST_SAND_v_0_1 folder.
2. Run the command "java -jar SAND_v_0_1.jar [OPTION(S)]"

### Option(s):
* -F 
 * Input file path. Do not include file extension .arff in the file path.
 
### Optional option(s):
* -S
  * Size of warm-up period chunks. Default size is 2000 instances.
* -L
  * Maximum number of models in the ensemble. Default value is 6.
* -U
  * Value for confidence threshold. Default value is 0.90. Please refer to the paper for description of confidence threshold.
* -D
  * use 1 here to execute SAND-D, 0 to execute SAND-F. Default value is 1. Please refer to the paper for description about SAND-D, and SAND-F.
* -T
  * Labeling delay in number of instances. Default value for classification only is 1. Use appropriate value for novel class detection.
* -C
  * Classification delay in number of instances. Default value for classification only is 0. Use appropriate value for novel class detection.


## Output
### Console output
* Progress or any change point detected throughout execution. 
* At the end, it reports percentage of labeled data used.

### File output
1. .log file contains important debug information.
2. .tmpres file contains the error rates for each chunk.  There are six columns as follows:
  * Chunk #= The current chunk number. Each chunk contains 1000 instances.
  * FP= How many existing class instances misclassified as novel class in this chunk.
  * FN= How many novel class instances misclassified as existing class in this chunk.
  * NC= How many novel class instances are actually there in this chunk.
  * Err = How many instances are misclassified (including FP and FN) in this chunk.
  * GlobErr = % Err (cumulative) upto the current chunk.
3. .res file contains the summary result, i.e., the following error rates:
  * FP% = % of existing class instances misclassified as novel
  * FN% = % of novel class instances misclassified as existing class instances.
  * NC (total) = total number of (actual) novel class instances.
  * ERR% = % classification error (including FP, FN, and misclassification within existing class).

## Reference
[SAND: Semi-supervised Adaptive Novel Class Detection and Classification over Data Stream](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12335)
