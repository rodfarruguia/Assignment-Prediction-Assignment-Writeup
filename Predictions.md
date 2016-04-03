# prediction data training.
Rodrigo Farruguia  
April 2, 2016  

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data Proccessing 
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

downloading the data


```r
knitr::opts_chunk$set(echo = TRUE)
if (!file.exists("trainfile")) {
    download.file(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
        destfile = "pml-training.csv")
}
if (!file.exists("testfile")) {
    download.file(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
        destfile = "pml-testing.csv")
}

trainfile <- "pml-training.csv"
testfile  <- "pml-testing.csv"
```
loading libraries


```r
library(rpart)#for regressive partitioning.
library(rpart.plot)# for decision tree plot.
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.4
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.4
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.2.4
```

```
## Loading required package: ggplot2
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 3.2.4
```

```r
library(manipulate)
library(xtable)
library(ggplot2)
library(lattice)
library(knitr)
```

```
## Warning: package 'knitr' was built under R version 3.2.4
```

```r
library(markdown)
```

```
## Warning: package 'markdown' was built under R version 3.2.4
```

```r
library(randomForest)# just in case i need to generate one for regression
```

```
## Warning: package 'randomForest' was built under R version 3.2.4
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```
###cleaning data, making it tidy.


```r
datatraining <- read.csv(trainfile,na.strings=c("NA","#DIV/0!",""))
datatesting <- read.csv(testfile,na.strings=c("NA","#DIV/0!",""))
```
taking out all the columns with missing values 


```r
datatraining <- datatraining[,colSums(is.na(datatraining)) == 0]
datatesting <- datatesting[,colSums(is.na(datatesting)) == 0]
```
taking out all variables that are not of interest

```r
datatraining <- datatraining[,-c(1:7)]
datatesting <- datatesting[,-c(1:7)]
```

the dimmensions of the data


```r
dim(datatraining)
```

```
## [1] 19622    53
```

```r
dim(datatesting)
```

```
## [1] 20 53
```

```r
table(datatraining$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

#Partitioning to allow cross-validation
The training data set is partionned into 2 sets: subdatatraining 60% and subdatatesting 40%.



```r
set.seed(1)#for reproducability
subsamples <- createDataPartition(y=datatraining$classe, p=0.60, list=FALSE)
subdatatraining <- datatraining[subsamples, ] 
subdatatesting <- datatraining[-subsamples, ]
dim(subdatatraining)
```

```
## [1] 11776    53
```

```r
dim(subdatatesting)
```

```
## [1] 7846   53
```

```r
head(subdatatraining)
```

```
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## 2      1.41       8.07    -94.4                3         0.02         0.00
## 3      1.42       8.07    -94.4                3         0.00         0.00
## 4      1.48       8.05    -94.4                3         0.02         0.00
## 5      1.48       8.07    -94.4                3         0.02         0.02
## 7      1.42       8.09    -94.4                3         0.02         0.00
## 8      1.42       8.13    -94.4                3         0.02         0.00
##   gyros_belt_z accel_belt_x accel_belt_y accel_belt_z magnet_belt_x
## 2        -0.02          -22            4           22            -7
## 3        -0.02          -20            5           23            -2
## 4        -0.03          -22            3           21            -6
## 5        -0.02          -21            2           24            -6
## 7        -0.02          -22            3           21            -4
## 8        -0.02          -22            4           21            -2
##   magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm
## 2           608          -311     -128      22.5    -161              34
## 3           600          -305     -128      22.5    -161              34
## 4           604          -310     -128      22.1    -161              34
## 5           600          -302     -128      22.1    -161              34
## 7           599          -311     -128      21.9    -161              34
## 8           603          -313     -128      21.8    -161              34
##   gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## 2        0.02       -0.02       -0.02        -290         110        -125
## 3        0.02       -0.02       -0.02        -289         110        -126
## 4        0.02       -0.03        0.02        -289         111        -123
## 5        0.00       -0.03        0.00        -289         111        -123
## 7        0.00       -0.03        0.00        -289         111        -125
## 8        0.02       -0.02        0.00        -289         111        -124
##   magnet_arm_x magnet_arm_y magnet_arm_z roll_dumbbell pitch_dumbbell
## 2         -369          337          513      13.13074      -70.63751
## 3         -368          344          513      12.85075      -70.27812
## 4         -372          344          512      13.43120      -70.39379
## 5         -374          337          506      13.37872      -70.42856
## 7         -373          336          509      13.12695      -70.24757
## 8         -372          338          510      12.75083      -70.34768
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 2    -84.71065                   37                0            -0.02
## 3    -85.14078                   37                0            -0.02
## 4    -84.87363                   37                0            -0.02
## 5    -84.85306                   37                0            -0.02
## 7    -85.09961                   37                0            -0.02
## 8    -85.09708                   37                0            -0.02
##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 2             0.00             -233               47             -269
## 3             0.00             -232               46             -270
## 4            -0.02             -232               48             -269
## 5             0.00             -233               48             -270
## 7             0.00             -232               47             -270
## 8             0.00             -234               46             -272
##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 2              -555               296               -64         28.3
## 3              -561               298               -63         28.3
## 4              -552               303               -60         28.1
## 5              -554               292               -68         28.0
## 7              -551               295               -70         27.9
## 8              -555               300               -74         27.8
##   pitch_forearm yaw_forearm total_accel_forearm gyros_forearm_x
## 2         -63.9        -153                  36            0.02
## 3         -63.9        -152                  36            0.03
## 4         -63.9        -152                  36            0.02
## 5         -63.9        -152                  36            0.02
## 7         -63.9        -152                  36            0.02
## 8         -63.8        -152                  36            0.02
##   gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
## 2            0.00           -0.02             192             203
## 3           -0.02            0.00             196             204
## 4           -0.02            0.00             189             206
## 5            0.00           -0.02             189             206
## 7            0.00           -0.02             195             205
## 8           -0.02            0.00             193             205
##   accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
## 2            -216              -18              661              473
## 3            -213              -18              658              469
## 4            -214              -16              658              469
## 5            -214              -17              655              473
## 7            -215              -18              659              470
## 8            -213               -9              660              474
##   classe
## 2      A
## 3      A
## 4      A
## 5      A
## 7      A
## 8      A
```

```r
head(subdatatesting)
```

```
##    roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x
## 1       1.41       8.07    -94.4                3         0.00
## 6       1.45       8.06    -94.4                3         0.02
## 12      1.43       8.18    -94.4                3         0.02
## 13      1.42       8.20    -94.4                3         0.02
## 15      1.45       8.20    -94.4                3         0.00
## 16      1.48       8.15    -94.4                3         0.00
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 1             0        -0.02          -21            4           22
## 6             0        -0.02          -21            4           21
## 12            0        -0.02          -22            2           23
## 13            0         0.00          -22            4           21
## 15            0         0.00          -21            2           22
## 16            0         0.00          -21            4           23
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 1             -3           599          -313     -128      22.5    -161
## 6              0           603          -312     -128      22.0    -161
## 12            -2           602          -319     -128      21.5    -161
## 13            -3           606          -309     -128      21.4    -161
## 15            -1           597          -310     -129      21.4    -161
## 16             0           592          -305     -129      21.3    -161
##    total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x
## 1               34        0.00        0.00       -0.02        -288
## 6               34        0.02       -0.03        0.00        -289
## 12              34        0.02       -0.03        0.00        -288
## 13              34        0.02       -0.02       -0.02        -287
## 15              34        0.02        0.00       -0.03        -289
## 16              34        0.02        0.00       -0.03        -289
##    accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 1          109        -123         -368          337          516
## 6          111        -122         -369          342          513
## 12         111        -123         -363          343          520
## 13         111        -124         -372          338          509
## 15         111        -124         -374          342          510
## 16         109        -121         -367          340          509
##    roll_dumbbell pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## 1       13.05217      -70.49400    -84.87394                   37
## 6       13.38246      -70.81759    -84.46500                   37
## 12      13.10321      -70.45975    -84.89472                   37
## 13      13.38246      -70.81759    -84.46500                   37
## 15      13.07949      -70.67116    -84.69053                   37
## 16      13.35069      -70.25176    -85.03639                   37
##    gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
## 1                 0            -0.02             0.00             -234
## 6                 0            -0.02             0.00             -234
## 12                0            -0.02             0.00             -233
## 13                0            -0.02            -0.02             -234
## 15                0            -0.02             0.00             -234
## 16                0            -0.02             0.00             -233
##    accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
## 1                47             -271              -559               293
## 6                48             -269              -558               294
## 12               47             -270              -554               291
## 13               48             -269              -552               302
## 15               47             -270              -554               294
## 16               48             -271              -554               297
##    magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 1                -65         28.4         -63.9        -153
## 6                -66         27.9         -63.9        -152
## 12               -65         27.5         -63.8        -152
## 13               -69         27.2         -63.9        -151
## 15               -63         27.2         -63.9        -151
## 16               -73         27.1         -64.0        -151
##    total_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 1                   36            0.03            0.00           -0.02
## 6                   36            0.02           -0.02           -0.03
## 12                  36            0.02            0.02           -0.03
## 13                  36            0.00            0.00           -0.03
## 15                  36            0.00           -0.02           -0.02
## 16                  36            0.02            0.00            0.00
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 1              192             203            -215              -17
## 6              193             203            -215               -9
## 12             191             203            -215              -11
## 13             193             205            -215              -15
## 15             192             201            -214              -16
## 16             194             204            -215              -13
##    magnet_forearm_y magnet_forearm_z classe
## 1               654              476      A
## 6               660              478      A
## 12              657              478      A
## 13              655              472      A
## 15              656              472      A
## 16              656              471      A
```

lets see if any patterns emerge! using correlation.



```r
corrPlot <- cor(subdatatraining[, -53])
plot1 <- corrplot(corrPlot, method="color")
```

![](Predictions_files/figure-html/unnamed-chunk-8-1.png)

```r
plot1
```

```
##                         roll_belt   pitch_belt     yaw_belt
## roll_belt             1.000000000 -0.220468892  0.814877911
## pitch_belt           -0.220468892  1.000000000 -0.700525707
## yaw_belt              0.814877911 -0.700525707  1.000000000
## total_accel_belt      0.980558237 -0.143786795  0.762005588
## gyros_belt_x         -0.110592029 -0.436622337  0.150338055
## gyros_belt_y          0.469176081 -0.401517040  0.533999759
## gyros_belt_z         -0.460456085 -0.108094262 -0.275079440
## accel_belt_x          0.260257821 -0.965575822  0.709048476
## accel_belt_y          0.925519955  0.076529312  0.600764843
## accel_belt_z         -0.992014313  0.166662076 -0.776367971
## magnet_belt_x         0.353644769 -0.882263492  0.732216910
## magnet_belt_y        -0.214153249 -0.085938784 -0.061269064
## magnet_belt_z        -0.068410910 -0.128619122  0.087184803
## roll_arm             -0.372696399  0.172865263 -0.337520370
## pitch_arm             0.059055789 -0.158532396  0.134324239
## yaw_arm              -0.223488128  0.145363684 -0.226720851
## total_accel_arm      -0.277878562  0.093799831 -0.241082613
## gyros_arm_x           0.031395530 -0.062494880  0.051173392
## gyros_arm_y          -0.228400271  0.118206254 -0.220805891
## gyros_arm_z           0.546728817 -0.310153211  0.549012201
## accel_arm_x           0.225202443  0.190223660  0.061734481
## accel_arm_y          -0.795534832  0.227355138 -0.677422150
## accel_arm_z           0.385761623  0.047230179  0.244174105
## magnet_arm_x          0.083983316 -0.047701054  0.081721916
## magnet_arm_y          0.008458036  0.047886154 -0.017194905
## magnet_arm_z          0.022114420  0.005998808  0.012473058
## roll_dumbbell        -0.130470725 -0.346775165  0.092142166
## pitch_dumbbell        0.060594259  0.233915277 -0.096589959
## yaw_dumbbell          0.030779098  0.661046288 -0.344186887
## total_accel_dumbbell -0.200753544 -0.311256480  0.036220048
## gyros_dumbbell_x      0.008707334  0.009462198  0.002886019
## gyros_dumbbell_y      0.050763305 -0.021878916  0.038731889
## gyros_dumbbell_z     -0.000885959 -0.005304017  0.001663410
## accel_dumbbell_x      0.215038108  0.167738007  0.044369648
## accel_dumbbell_y     -0.272598293 -0.443817856  0.043705281
## accel_dumbbell_z      0.107457301  0.537152776 -0.227479986
## magnet_dumbbell_x     0.304010080  0.463610624 -0.036077754
## magnet_dumbbell_y    -0.286971036 -0.355773062 -0.036466197
## magnet_dumbbell_z    -0.501516005 -0.254556348 -0.217126645
## roll_forearm         -0.149274616  0.146067842 -0.181037901
## pitch_forearm         0.176018420  0.255390147 -0.031155763
## yaw_forearm          -0.269135128  0.046977325 -0.196304276
## total_accel_forearm   0.077820127 -0.330906214  0.238304889
## gyros_forearm_x       0.377596639 -0.529386526  0.561597909
## gyros_forearm_y       0.034141137 -0.057841442  0.060778380
## gyros_forearm_z       0.015370833 -0.075234577  0.054069084
## accel_forearm_x      -0.494602490  0.129248588 -0.399430467
## accel_forearm_y       0.030049348 -0.366305475  0.240357900
## accel_forearm_z       0.083836359 -0.229668266  0.172420735
## magnet_forearm_x     -0.202898875 -0.066197442 -0.101369131
## magnet_forearm_y      0.025282893 -0.018887667  0.043467517
## magnet_forearm_z      0.272236474 -0.071160028  0.231730193
##                      total_accel_belt gyros_belt_x gyros_belt_y
## roll_belt                 0.980558237 -0.110592029  0.469176081
## pitch_belt               -0.143786795 -0.436622337 -0.401517040
## yaw_belt                  0.762005588  0.150338055  0.533999759
## total_accel_belt          1.000000000 -0.158730370  0.414806764
## gyros_belt_x             -0.158730370  1.000000000  0.338691304
## gyros_belt_y              0.414806764  0.338691304  1.000000000
## gyros_belt_z             -0.475642316  0.331119008  0.329793648
## accel_belt_x              0.175536652  0.471966052  0.451752329
## accel_belt_y              0.928459024 -0.262282704  0.333496188
## accel_belt_z             -0.974548394  0.145496960 -0.438429161
## magnet_belt_x             0.288907127  0.432108025  0.399493136
## magnet_belt_y            -0.211952071  0.105709167  0.063385569
## magnet_belt_z            -0.054719942  0.237132485  0.117045817
## roll_arm                 -0.361305107 -0.243899367 -0.377669278
## pitch_arm                 0.052583157 -0.026961765  0.008542203
## yaw_arm                  -0.206357044 -0.150859479 -0.238242647
## total_accel_arm          -0.267169508  0.001996391 -0.156334696
## gyros_arm_x               0.022115651  0.067000422  0.087827866
## gyros_arm_y              -0.216607299 -0.054595629 -0.166183533
## gyros_arm_z               0.512531806  0.280568461  0.487331752
## accel_arm_x               0.245541383 -0.372651855 -0.125886240
## accel_arm_y              -0.758952495 -0.081422397 -0.495793782
## accel_arm_z               0.391606346 -0.255486355  0.051374133
## magnet_arm_x              0.077052591  0.026068123  0.082381287
## magnet_arm_y              0.034248950 -0.134368973 -0.090032810
## magnet_arm_z              0.020675889 -0.017414665  0.010701387
## roll_dumbbell            -0.134259118  0.256086447  0.065996847
## pitch_dumbbell            0.056281042 -0.188299576 -0.130603218
## yaw_dumbbell              0.066900609 -0.433050717 -0.288342263
## total_accel_dumbbell     -0.179742009  0.059940698 -0.073757310
## gyros_dumbbell_x          0.020173318 -0.055631979 -0.017413797
## gyros_dumbbell_y          0.044839962  0.101501534  0.036961319
## gyros_dumbbell_z         -0.005386443  0.016663308  0.006515794
## accel_dumbbell_x          0.186151009 -0.126385570  0.013374033
## accel_dumbbell_y         -0.271607707  0.295689475  0.041347100
## accel_dumbbell_z          0.115478536 -0.348709159 -0.184073100
## magnet_dumbbell_x         0.355900079 -0.741555403 -0.347820575
## magnet_dumbbell_y        -0.336831365  0.775967822  0.374106251
## magnet_dumbbell_z        -0.516045369  0.224330678 -0.169567215
## roll_forearm             -0.112485861 -0.226174601 -0.219899636
## pitch_forearm             0.185987267 -0.157013562 -0.016128360
## yaw_forearm              -0.239711765 -0.184873580 -0.277436736
## total_accel_forearm       0.046197621  0.065101377  0.078356846
## gyros_forearm_x           0.347139709  0.129853571  0.310077701
## gyros_forearm_y           0.024594034  0.021854227 -0.019147385
## gyros_forearm_z           0.006420906  0.014185762  0.009666955
## accel_forearm_x          -0.456884579 -0.219437198 -0.391729159
## accel_forearm_y           0.019932474 -0.041354639 -0.021491785
## accel_forearm_z           0.046196173  0.347881453  0.262334184
## magnet_forearm_x         -0.197231811  0.044173474 -0.042777655
## magnet_forearm_y          0.036895492  0.018245599  0.004831068
## magnet_forearm_z          0.295290478 -0.112290313  0.121738819
##                       gyros_belt_z accel_belt_x accel_belt_y  accel_belt_z
## roll_belt            -0.4604560854  0.260257821  0.925519955 -0.9920143129
## pitch_belt           -0.1080942619 -0.965575822  0.076529312  0.1666620758
## yaw_belt             -0.2750794400  0.709048476  0.600764843 -0.7763679712
## total_accel_belt     -0.4756423165  0.175536652  0.928459024 -0.9745483936
## gyros_belt_x          0.3311190083  0.471966052 -0.262282704  0.1454969596
## gyros_belt_y          0.3297936484  0.451752329  0.333496188 -0.4384291606
## gyros_belt_z          1.0000000000  0.115235487 -0.511421758  0.4743572712
## accel_belt_x          0.1152354866  1.000000000 -0.034253232 -0.2129909151
## accel_belt_y         -0.5114217583 -0.034253232  1.000000000 -0.9339769769
## accel_belt_z          0.4743572712 -0.212990915 -0.933976977  1.0000000000
## magnet_belt_x         0.0258911760  0.889480554  0.088018780 -0.3032367150
## magnet_belt_y         0.1315258456  0.007342648 -0.198884110  0.2980346224
## magnet_belt_z         0.0785828341  0.071792917 -0.082971875  0.1356935991
## roll_arm              0.0359911051 -0.197877691 -0.300166377  0.3383714545
## pitch_arm            -0.0350465679  0.126959479  0.032137193 -0.0455600685
## yaw_arm               0.0131565552 -0.141839961 -0.178642433  0.2014428123
## total_accel_arm       0.0869263996 -0.097026550 -0.250777440  0.2826189943
## gyros_arm_x           0.0983423742  0.071168050  0.015558372 -0.0228538894
## gyros_arm_y           0.0072842698 -0.132301642 -0.192399037  0.2180969172
## gyros_arm_z          -0.1123316035  0.348981073  0.445537748 -0.5263832262
## accel_arm_x          -0.2679351280 -0.194370267  0.295248229 -0.2396501071
## accel_arm_y           0.3184996828 -0.265683264 -0.723104920  0.7872885401
## accel_arm_z          -0.2321167497 -0.053410551  0.410874769 -0.3958957993
## magnet_arm_x         -0.0173497704  0.048151041  0.064657588 -0.0806251886
## magnet_arm_y         -0.0366465069 -0.061916585  0.031827438  0.0005362126
## magnet_arm_z          0.0211814844 -0.013730136  0.033560226 -0.0235845647
## roll_dumbbell         0.0703017278  0.322602446 -0.237583317  0.1632593364
## pitch_dumbbell       -0.2169358159 -0.224385253  0.144790149 -0.1026688241
## yaw_dumbbell         -0.2439860758 -0.646700282  0.246868984 -0.0898002387
## total_accel_dumbbell  0.1194705751  0.275172668 -0.298955061  0.2580523148
## gyros_dumbbell_x     -0.0086094318 -0.016106243  0.011633599 -0.0061005155
## gyros_dumbbell_y     -0.0347359286  0.023886062  0.035470003 -0.0505605255
## gyros_dumbbell_z      0.0020186390  0.006632862 -0.001663699 -0.0006124785
## accel_dumbbell_x     -0.2236796651 -0.140863837  0.275158617 -0.2666579291
## accel_dumbbell_y      0.2010322845  0.416918306 -0.412123875  0.3308906125
## accel_dumbbell_z     -0.2468477288 -0.513229476  0.281775482 -0.1740501229
## magnet_dumbbell_x    -0.4961743700 -0.477700021  0.464118157 -0.3392946968
## magnet_dumbbell_y     0.4261086535  0.379211032 -0.423790646  0.3182552010
## magnet_dumbbell_z     0.2638161024  0.246940908 -0.583427996  0.5064619230
## roll_forearm          0.0021909321 -0.170758887 -0.104592573  0.1561553845
## pitch_forearm        -0.1583712556 -0.236742630  0.256056893 -0.2049517772
## yaw_forearm           0.0860202717 -0.078204696 -0.257095914  0.2788573194
## total_accel_forearm   0.0144150102  0.309017295 -0.025131778 -0.0600352851
## gyros_forearm_x      -0.1013104884  0.521577564  0.221841371 -0.3499339764
## gyros_forearm_y      -0.0220192267  0.056307245  0.014408597 -0.0283514044
## gyros_forearm_z       0.0004446853  0.074607839 -0.006676146 -0.0120702584
## accel_forearm_x       0.1394403705 -0.172389454 -0.451241079  0.4909043514
## accel_forearm_y      -0.0062677615  0.345141203 -0.078643140 -0.0202432997
## accel_forearm_z       0.0935208225  0.290135322  0.015658618 -0.0900771732
## magnet_forearm_x      0.1244580169  0.045179445 -0.220360504  0.2103401249
## magnet_forearm_y      0.0038891479  0.034688753  0.017309147 -0.0234661185
## magnet_forearm_z     -0.1133528075  0.046962441  0.245927666 -0.2400659818
##                      magnet_belt_x magnet_belt_y magnet_belt_z
## roll_belt              0.353644769  -0.214153249  -0.068410910
## pitch_belt            -0.882263492  -0.085938784  -0.128619122
## yaw_belt               0.732216910  -0.061269064   0.087184803
## total_accel_belt       0.288907127  -0.211952071  -0.054719942
## gyros_belt_x           0.432108025   0.105709167   0.237132485
## gyros_belt_y           0.399493136   0.063385569   0.117045817
## gyros_belt_z           0.025891176   0.131525846   0.078582834
## accel_belt_x           0.889480554   0.007342648   0.071792917
## accel_belt_y           0.088018780  -0.198884110  -0.082971875
## accel_belt_z          -0.303236715   0.298034622   0.135693599
## magnet_belt_x          1.000000000  -0.057356433   0.115756750
## magnet_belt_y         -0.057356433   1.000000000   0.764083690
## magnet_belt_z          0.115756750   0.764083690   1.000000000
## roll_arm              -0.209833820  -0.165410827  -0.211072301
## pitch_arm              0.174502679   0.034508939   0.043215575
## yaw_arm               -0.122635236  -0.158169244  -0.197092317
## total_accel_arm       -0.082454620   0.066295933   0.006754383
## gyros_arm_x            0.075290503   0.027232871   0.040931834
## gyros_arm_y           -0.156782148  -0.003923986  -0.062897969
## gyros_arm_z            0.381051698  -0.018968668   0.149923298
## accel_arm_x           -0.155379957  -0.114728102  -0.134623636
## accel_arm_y           -0.293295684   0.108374451  -0.052264774
## accel_arm_z           -0.003035550  -0.090955120  -0.040032382
## magnet_arm_x           0.010670761   0.018612506   0.018881425
## magnet_arm_y           0.005192696   0.090992045   0.076745610
## magnet_arm_z           0.001213464   0.028733628   0.034053657
## roll_dumbbell          0.284963955   0.227515949   0.245029297
## pitch_dumbbell        -0.256875989  -0.239538631  -0.170144250
## yaw_dumbbell          -0.605608887  -0.239246123  -0.201920515
## total_accel_dumbbell   0.326688667   0.368854403   0.208121807
## gyros_dumbbell_x      -0.009318540   0.019257153   0.008196658
## gyros_dumbbell_y       0.042544143  -0.024306337  -0.005026635
## gyros_dumbbell_z       0.005108660  -0.012243170  -0.008844498
## accel_dumbbell_x      -0.195736066  -0.370576613  -0.236705650
## accel_dumbbell_y       0.407541406   0.411453948   0.317319014
## accel_dumbbell_z      -0.500677173  -0.409987989  -0.347194297
## magnet_dumbbell_x     -0.353385006  -0.244245743  -0.285877617
## magnet_dumbbell_y      0.247131691   0.231126073   0.253048051
## magnet_dumbbell_z      0.222130258  -0.183522291  -0.299468735
## roll_forearm          -0.158309710   0.156488959   0.050781715
## pitch_forearm         -0.243850818  -0.128054116  -0.091923699
## yaw_forearm           -0.022633076   0.082469667   0.011397044
## total_accel_forearm    0.259803034   0.006138411   0.062649577
## gyros_forearm_x        0.537026032   0.009722665   0.115510013
## gyros_forearm_y        0.055726968   0.033595047   0.037187443
## gyros_forearm_z        0.072546838  -0.009181799  -0.013680506
## accel_forearm_x       -0.174166567   0.070903933  -0.085317236
## accel_forearm_y        0.332599591  -0.038625920  -0.007287379
## accel_forearm_z        0.298215608  -0.355316876  -0.362546062
## magnet_forearm_x       0.021975324   0.015668218  -0.089072663
## magnet_forearm_y       0.075950970   0.004548481   0.042000279
## magnet_forearm_z       0.047325020   0.379616801   0.393219256
##                          roll_arm    pitch_arm       yaw_arm
## roll_belt            -0.372696399  0.059055789 -0.2234881284
## pitch_belt            0.172865263 -0.158532396  0.1453636844
## yaw_belt             -0.337520370  0.134324239 -0.2267208508
## total_accel_belt     -0.361305107  0.052583157 -0.2063570437
## gyros_belt_x         -0.243899367 -0.026961765 -0.1508594788
## gyros_belt_y         -0.377669278  0.008542203 -0.2382426466
## gyros_belt_z          0.035991105 -0.035046568  0.0131565552
## accel_belt_x         -0.197877691  0.126959479 -0.1418399611
## accel_belt_y         -0.300166377  0.032137193 -0.1786424335
## accel_belt_z          0.338371454 -0.045560068  0.2014428123
## magnet_belt_x        -0.209833820  0.174502679 -0.1226352361
## magnet_belt_y        -0.165410827  0.034508939 -0.1581692438
## magnet_belt_z        -0.211072301  0.043215575 -0.1970923175
## roll_arm              1.000000000 -0.039343202  0.4001853940
## pitch_arm            -0.039343202  1.000000000 -0.0916976295
## yaw_arm               0.400185394 -0.091697630  1.0000000000
## total_accel_arm       0.053018248  0.028637847 -0.0184792785
## gyros_arm_x          -0.021363135  0.022204155 -0.0130903554
## gyros_arm_y           0.092541543  0.007946074  0.0758948713
## gyros_arm_z          -0.346586220  0.001610531 -0.2378037743
## accel_arm_x           0.166244924 -0.479666284  0.0561703199
## accel_arm_y           0.340036203  0.188890007  0.3004224332
## accel_arm_z          -0.062396219  0.321689507  0.0268627759
## magnet_arm_x          0.049452335 -0.534909252 -0.0240412576
## magnet_arm_y         -0.117875159  0.553759574  0.0639945416
## magnet_arm_z         -0.088983522  0.527294343  0.0291748432
## roll_dumbbell        -0.154539901  0.029579900 -0.1223819194
## pitch_dumbbell        0.226089103 -0.148339896  0.0881732245
## yaw_dumbbell          0.273582085 -0.173522639  0.1471420130
## total_accel_dumbbell -0.151129146  0.192127538 -0.0573793980
## gyros_dumbbell_x      0.005832367  0.004565596 -0.0003540145
## gyros_dumbbell_y     -0.109269254  0.008586829 -0.0793515471
## gyros_dumbbell_z      0.005759552  0.006724298  0.0122956454
## accel_dumbbell_x      0.252345802 -0.178373309  0.1145120233
## accel_dumbbell_y     -0.169710700  0.073128576 -0.0843648230
## accel_dumbbell_z      0.340153486 -0.129997042  0.2068213583
## magnet_dumbbell_x     0.215407173 -0.001722770  0.1389207359
## magnet_dumbbell_y    -0.260304366 -0.055533500 -0.1567696034
## magnet_dumbbell_z     0.358315717  0.011149811  0.2443452495
## roll_forearm          0.074416696  0.022917953  0.0618560461
## pitch_forearm         0.167248804 -0.152617326  0.0630956300
## yaw_forearm           0.234868082  0.081323519  0.1905575010
## total_accel_forearm  -0.029409455  0.103965645 -0.1134378913
## gyros_forearm_x      -0.170529379  0.110962223 -0.1057118339
## gyros_forearm_y      -0.024502207  0.021417538 -0.0263076641
## gyros_forearm_z       0.013144770  0.028706250  0.0046760966
## accel_forearm_x       0.226344597  0.021328742  0.1600330861
## accel_forearm_y       0.064847841  0.093315025  0.0289773879
## accel_forearm_z       0.075433324 -0.002018454  0.0884494404
## magnet_forearm_x      0.056887289  0.012677203  0.0508361631
## magnet_forearm_y     -0.094966389  0.055419612 -0.0204731811
## magnet_forearm_z     -0.337649202 -0.026043777 -0.2351422014
##                      total_accel_arm   gyros_arm_x   gyros_arm_y
## roll_belt               -0.277878562  0.0313955296 -0.2284002713
## pitch_belt               0.093799831 -0.0624948798  0.1182062541
## yaw_belt                -0.241082613  0.0511733920 -0.2208058911
## total_accel_belt        -0.267169508  0.0221156505 -0.2166072995
## gyros_belt_x             0.001996391  0.0670004216 -0.0545956291
## gyros_belt_y            -0.156334696  0.0878278663 -0.1661835335
## gyros_belt_z             0.086926400  0.0983423742  0.0072842698
## accel_belt_x            -0.097026550  0.0711680500 -0.1323016418
## accel_belt_y            -0.250777440  0.0155583716 -0.1923990369
## accel_belt_z             0.282618994 -0.0228538894  0.2180969172
## magnet_belt_x           -0.082454620  0.0752905032 -0.1567821482
## magnet_belt_y            0.066295933  0.0272328711 -0.0039239860
## magnet_belt_z            0.006754383  0.0409318341 -0.0628979693
## roll_arm                 0.053018248 -0.0213631352  0.0925415434
## pitch_arm                0.028637847  0.0222041550  0.0079460737
## yaw_arm                 -0.018479279 -0.0130903554  0.0758948713
## total_accel_arm          1.000000000 -0.0162223683  0.0769350276
## gyros_arm_x             -0.016222368  1.0000000000 -0.9182301260
## gyros_arm_y              0.076935028 -0.9182301260  1.0000000000
## gyros_arm_z             -0.101325520  0.3382655023 -0.4812982012
## accel_arm_x             -0.106048601 -0.0220860759 -0.0613113936
## accel_arm_y              0.087097310 -0.0246181814  0.2026294415
## accel_arm_z             -0.610410250 -0.0468832481 -0.0178614033
## magnet_arm_x            -0.263692028 -0.0266275769 -0.0635630637
## magnet_arm_y            -0.046416167  0.0722668956 -0.0167793458
## magnet_arm_z            -0.377208843  0.0255683247  0.0144341532
## roll_dumbbell           -0.002271651  0.0150383862 -0.0084144307
## pitch_dumbbell          -0.072757259 -0.0408220784  0.0172009454
## yaw_dumbbell            -0.033333059 -0.0670067599  0.0628465060
## total_accel_dumbbell     0.138786486  0.0173858836  0.0225540028
## gyros_dumbbell_x         0.001638303 -0.0169299516  0.0116547949
## gyros_dumbbell_y         0.041594012 -0.0019980526  0.0006968087
## gyros_dumbbell_z        -0.003348720  0.0026870961  0.0005772699
## accel_dumbbell_x        -0.118251954 -0.0199648907 -0.0328765087
## accel_dumbbell_y         0.109810917  0.0217592757  0.0171720853
## accel_dumbbell_z        -0.087054468 -0.0446716864  0.0280883432
## magnet_dumbbell_x       -0.046226451 -0.0750459897  0.0299377248
## magnet_dumbbell_y        0.035757270  0.0661254039 -0.0278510504
## magnet_dumbbell_z        0.134259442  0.0044382659  0.0988464057
## roll_forearm             0.005111745 -0.0174252332  0.0357602663
## pitch_forearm           -0.183202532 -0.0758262899  0.0366024300
## yaw_forearm              0.079662326  0.0514685323  0.0113379041
## total_accel_forearm     -0.076805225  0.0476646808 -0.0596449245
## gyros_forearm_x         -0.169395176  0.0401189120 -0.1151515592
## gyros_forearm_y          0.027288120 -0.0190502065 -0.0052749228
## gyros_forearm_z          0.016569236  0.0009575252 -0.0064623833
## accel_forearm_x          0.245074915 -0.0422771193  0.1426099054
## accel_forearm_y         -0.045252363  0.1125608198 -0.1171587407
## accel_forearm_z         -0.057175544  0.0613444311 -0.0760659305
## magnet_forearm_x         0.157518570 -0.0653678523  0.0960792372
## magnet_forearm_y        -0.006059733  0.1661918130 -0.1522186627
## magnet_forearm_z        -0.097250422  0.0136659441 -0.0981647687
##                       gyros_arm_z   accel_arm_x  accel_arm_y   accel_arm_z
## roll_belt             0.546728817  0.2252024429 -0.795534832  0.3857616233
## pitch_belt           -0.310153211  0.1902236599  0.227355138  0.0472301788
## yaw_belt              0.549012201  0.0617344814 -0.677422150  0.2441741050
## total_accel_belt      0.512531806  0.2455413831 -0.758952495  0.3916063464
## gyros_belt_x          0.280568461 -0.3726518549 -0.081422397 -0.2554863546
## gyros_belt_y          0.487331752 -0.1258862404 -0.495793782  0.0513741333
## gyros_belt_z         -0.112331604 -0.2679351280  0.318499683 -0.2321167497
## accel_belt_x          0.348981073 -0.1943702675 -0.265683264 -0.0534105510
## accel_belt_y          0.445537748  0.2952482287 -0.723104920  0.4108747686
## accel_belt_z         -0.526383226 -0.2396501071  0.787288540 -0.3958957993
## magnet_belt_x         0.381051698 -0.1553799565 -0.293295684 -0.0030355498
## magnet_belt_y        -0.018968668 -0.1147281020  0.108374451 -0.0909551198
## magnet_belt_z         0.149923298 -0.1346236355 -0.052264774 -0.0400323823
## roll_arm             -0.346586220  0.1662449240  0.340036203 -0.0623962191
## pitch_arm             0.001610531 -0.4796662840  0.188890007  0.3216895072
## yaw_arm              -0.237803774  0.0561703199  0.300422433  0.0268627759
## total_accel_arm      -0.101325520 -0.1060486013  0.087097310 -0.6104102496
## gyros_arm_x           0.338265502 -0.0220860759 -0.024618181 -0.0468832481
## gyros_arm_y          -0.481298201 -0.0613113936  0.202629441 -0.0178614033
## gyros_arm_z           1.000000000 -0.0080714749 -0.587936757  0.0038411642
## accel_arm_x          -0.008071475  1.0000000000 -0.415520907 -0.1519500981
## accel_arm_y          -0.587936757 -0.4155209068  1.000000000  0.1259302827
## accel_arm_z           0.003841164 -0.1519500981  0.125930283  1.0000000000
## magnet_arm_x          0.078264594  0.8148576837 -0.338395239 -0.1530734041
## magnet_arm_y         -0.106233053 -0.7035251684  0.488372882  0.5561234730
## magnet_arm_z         -0.062964664 -0.6594312307  0.424988337  0.7820795777
## roll_dumbbell         0.043231928 -0.0803129325  0.060420237 -0.1149710403
## pitch_dumbbell       -0.056258953  0.2444716779 -0.084687140  0.0955056044
## yaw_dumbbell         -0.177883698  0.3152217520 -0.017077015  0.1556081870
## total_accel_dumbbell -0.096296114 -0.1714291671  0.278980440 -0.1206143016
## gyros_dumbbell_x     -0.025002916  0.0325900293  0.012545403  0.0175831320
## gyros_dumbbell_y      0.073758779 -0.0508901908 -0.066077897 -0.0398407132
## gyros_dumbbell_z      0.004681008 -0.0149792104 -0.005291446 -0.0004875015
## accel_dumbbell_x      0.064471991  0.2330179504 -0.249529280  0.1299345040
## accel_dumbbell_y     -0.019771883 -0.2061621177  0.235492391 -0.2089697148
## accel_dumbbell_z     -0.099609768  0.2967848168 -0.101063783  0.1705641938
## magnet_dumbbell_x    -0.181406363  0.4818774509 -0.080109508  0.3143362684
## magnet_dumbbell_y     0.213952686 -0.4197009797  0.039087750 -0.3303263295
## magnet_dumbbell_z    -0.225933121 -0.1015667366  0.432766770 -0.2653390295
## roll_forearm         -0.197909266  0.0575267251  0.194030268  0.0117419155
## pitch_forearm         0.023361870  0.3751407442 -0.207473055  0.1489578076
## yaw_forearm          -0.248064785 -0.0351612232  0.323927259 -0.0416583600
## total_accel_forearm   0.077494219  0.0024454298 -0.061887458  0.0519164577
## gyros_forearm_x       0.259459021 -0.0594805269 -0.270871668  0.1661068074
## gyros_forearm_y       0.012625800  0.0028476535 -0.051885290 -0.0245456527
## gyros_forearm_z       0.011162309 -0.0007257841 -0.018966109 -0.0177676621
## accel_forearm_x      -0.427429735 -0.1422933447  0.522565065 -0.1583774774
## accel_forearm_y      -0.003262478 -0.0448578365  0.036536955  0.0518822500
## accel_forearm_z       0.203143815 -0.0700194534 -0.104493211 -0.0700754817
## magnet_forearm_x     -0.135693644 -0.1835996048  0.195745565 -0.1338366139
## magnet_forearm_y      0.057206634 -0.1378472741  0.028325771  0.0381496410
## magnet_forearm_z      0.126892516  0.0445893833 -0.222749835  0.1021260369
##                      magnet_arm_x  magnet_arm_y magnet_arm_z roll_dumbbell
## roll_belt             0.083983316  0.0084580363  0.022114420 -0.1304707247
## pitch_belt           -0.047701054  0.0478861541  0.005998808 -0.3467751647
## yaw_belt              0.081721916 -0.0171949045  0.012473058  0.0921421664
## total_accel_belt      0.077052591  0.0342489499  0.020675889 -0.1342591182
## gyros_belt_x          0.026068123 -0.1343689732 -0.017414665  0.2560864473
## gyros_belt_y          0.082381287 -0.0900328102  0.010701387  0.0659968473
## gyros_belt_z         -0.017349770 -0.0366465069  0.021181484  0.0703017278
## accel_belt_x          0.048151041 -0.0619165853 -0.013730136  0.3226024464
## accel_belt_y          0.064657588  0.0318274381  0.033560226 -0.2375833173
## accel_belt_z         -0.080625189  0.0005362126 -0.023584565  0.1632593364
## magnet_belt_x         0.010670761  0.0051926957  0.001213464  0.2849639550
## magnet_belt_y         0.018612506  0.0909920448  0.028733628  0.2275159489
## magnet_belt_z         0.018881425  0.0767456103  0.034053657  0.2450292969
## roll_arm              0.049452335 -0.1178751587 -0.088983522 -0.1545399011
## pitch_arm            -0.534909252  0.5537595739  0.527294343  0.0295798996
## yaw_arm              -0.024041258  0.0639945416  0.029174843 -0.1223819194
## total_accel_arm      -0.263692028 -0.0464161670 -0.377208843 -0.0022716506
## gyros_arm_x          -0.026627577  0.0722668956  0.025568325  0.0150383862
## gyros_arm_y          -0.063563064 -0.0167793458  0.014434153 -0.0084144307
## gyros_arm_z           0.078264594 -0.1062330530 -0.062964664  0.0432319282
## accel_arm_x           0.814857684 -0.7035251684 -0.659431231 -0.0803129325
## accel_arm_y          -0.338395239  0.4883728818  0.424988337  0.0604202373
## accel_arm_z          -0.153073404  0.5561234730  0.782079578 -0.1149710403
## magnet_arm_x          1.000000000 -0.7909155770 -0.580139632  0.0911992439
## magnet_arm_y         -0.790915577  1.0000000000  0.816003503 -0.0204493957
## magnet_arm_z         -0.580139632  0.8160035031  1.000000000 -0.0432186960
## roll_dumbbell         0.091199244 -0.0204493957 -0.043218696  1.0000000000
## pitch_dumbbell        0.116430539 -0.1422559353 -0.070753870  0.2157773500
## yaw_dumbbell          0.048127710 -0.0907245733 -0.036059029 -0.2774773745
## total_accel_dumbbell -0.099922553  0.2139469547  0.031259896  0.3583577084
## gyros_dumbbell_x      0.008684100  0.0179764191 -0.006895154 -0.0030567638
## gyros_dumbbell_y     -0.013010736 -0.0063638530 -0.018943588  0.0312552328
## gyros_dumbbell_z     -0.010423015 -0.0048750596  0.010919809  0.0004172684
## accel_dumbbell_x      0.120624996 -0.2076828859 -0.074127427 -0.0706797668
## accel_dumbbell_y      0.016498886  0.0832169488 -0.034340837  0.7183893624
## accel_dumbbell_z      0.057216196 -0.1633863455 -0.022564024 -0.3218448161
## magnet_dumbbell_x     0.035633056  0.0750508861 -0.029410707 -0.1586164885
## magnet_dumbbell_y     0.049446424 -0.1461663552 -0.027143524  0.4112970236
## magnet_dumbbell_z     0.009248756 -0.1299308441 -0.088707005  0.1267696997
## roll_forearm         -0.020374750  0.0954512131  0.014032625 -0.0212644816
## pitch_forearm         0.339628117 -0.2697038747 -0.132219908 -0.1075936493
## yaw_forearm          -0.135619629  0.1527817154  0.051896242 -0.0011980787
## total_accel_forearm   0.040449887 -0.0182111952  0.008255227  0.1810865459
## gyros_forearm_x      -0.015024442  0.0862568047  0.089528538  0.1698336442
## gyros_forearm_y       0.013568497 -0.0307675229 -0.021478866  0.0041289650
## gyros_forearm_z       0.002615819 -0.0193419093 -0.012810644  0.0203444762
## accel_forearm_x      -0.231677899  0.2017121384  0.041525036  0.0028203545
## accel_forearm_y      -0.078722545  0.0755653978  0.069971115  0.0841814208
## accel_forearm_z       0.032248529 -0.1740004580 -0.030933766 -0.0668224802
## magnet_forearm_x     -0.152978048  0.0919217299  0.020327580  0.0129928690
## magnet_forearm_y     -0.169736404  0.1508319446  0.132649904 -0.0315732828
## magnet_forearm_z      0.050102420  0.1199374222  0.005862917  0.1514855791
##                      pitch_dumbbell yaw_dumbbell total_accel_dumbbell
## roll_belt               0.060594259   0.03077910          -0.20075354
## pitch_belt              0.233915277   0.66104629          -0.31125648
## yaw_belt               -0.096589959  -0.34418689           0.03622005
## total_accel_belt        0.056281042   0.06690061          -0.17974201
## gyros_belt_x           -0.188299576  -0.43305072           0.05994070
## gyros_belt_y           -0.130603218  -0.28834226          -0.07375731
## gyros_belt_z           -0.216935816  -0.24398608           0.11947058
## accel_belt_x           -0.224385253  -0.64670028           0.27517267
## accel_belt_y            0.144790149   0.24686898          -0.29895506
## accel_belt_z           -0.102668824  -0.08980024           0.25805231
## magnet_belt_x          -0.256875989  -0.60560889           0.32668867
## magnet_belt_y          -0.239538631  -0.23924612           0.36885440
## magnet_belt_z          -0.170144250  -0.20192052           0.20812181
## roll_arm                0.226089103   0.27358209          -0.15112915
## pitch_arm              -0.148339896  -0.17352264           0.19212754
## yaw_arm                 0.088173224   0.14714201          -0.05737940
## total_accel_arm        -0.072757259  -0.03333306           0.13878649
## gyros_arm_x            -0.040822078  -0.06700676           0.01738588
## gyros_arm_y             0.017200945   0.06284651           0.02255400
## gyros_arm_z            -0.056258953  -0.17788370          -0.09629611
## accel_arm_x             0.244471678   0.31522175          -0.17142917
## accel_arm_y            -0.084687140  -0.01707701           0.27898044
## accel_arm_z             0.095505604   0.15560819          -0.12061430
## magnet_arm_x            0.116430539   0.04812771          -0.09992255
## magnet_arm_y           -0.142255935  -0.09072457           0.21394695
## magnet_arm_z           -0.070753870  -0.03605903           0.03125990
## roll_dumbbell           0.215777350  -0.27747737           0.35835771
## pitch_dumbbell          1.000000000   0.51610777          -0.40287280
## yaw_dumbbell            0.516107766   1.00000000          -0.52352830
## total_accel_dumbbell   -0.402872805  -0.52352830           1.00000000
## gyros_dumbbell_x        0.004675365  -0.01257494           0.00320183
## gyros_dumbbell_y       -0.100304523  -0.05944437           0.08131294
## gyros_dumbbell_z       -0.004869957   0.01528007           0.01115439
## accel_dumbbell_x        0.810284410   0.54157697          -0.67579724
## accel_dumbbell_y       -0.159022846  -0.54049170           0.77319540
## accel_dumbbell_z        0.574466784   0.84761071          -0.62580655
## magnet_dumbbell_x       0.490989664   0.57747565          -0.10456958
## magnet_dumbbell_y      -0.119613454  -0.43781113           0.16079419
## magnet_dumbbell_z       0.068137453  -0.11506132           0.14955018
## roll_forearm           -0.074561975   0.01282051           0.23406891
## pitch_forearm           0.293212822   0.28355067          -0.33846508
## yaw_forearm            -0.101232618  -0.06458712           0.23992731
## total_accel_forearm     0.006826758  -0.19530236           0.11933947
## gyros_forearm_x        -0.094547527  -0.31107361           0.13072442
## gyros_forearm_y        -0.030011408  -0.03200615           0.06917671
## gyros_forearm_z        -0.017640712  -0.03202320           0.06147762
## accel_forearm_x        -0.082353669  -0.01592940           0.26650444
## accel_forearm_y        -0.099861020  -0.19910195           0.18337294
## accel_forearm_z         0.125451113  -0.01719931          -0.25070858
## magnet_forearm_x       -0.026269098  -0.09451502           0.08100293
## magnet_forearm_y       -0.154951515  -0.06682397           0.09307803
## magnet_forearm_z       -0.216918057  -0.19641840           0.30513271
##                      gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z
## roll_belt                0.0087073341     0.0507633048    -0.0008859590
## pitch_belt               0.0094621982    -0.0218789155    -0.0053040171
## yaw_belt                 0.0028860189     0.0387318885     0.0016634100
## total_accel_belt         0.0201733183     0.0448399620    -0.0053864432
## gyros_belt_x            -0.0556319794     0.1015015344     0.0166633082
## gyros_belt_y            -0.0174137966     0.0369613194     0.0065157940
## gyros_belt_z            -0.0086094318    -0.0347359286     0.0020186390
## accel_belt_x            -0.0161062433     0.0238860618     0.0066328616
## accel_belt_y             0.0116335985     0.0354700026    -0.0016636991
## accel_belt_z            -0.0061005155    -0.0505605255    -0.0006124785
## magnet_belt_x           -0.0093185400     0.0425441434     0.0051086604
## magnet_belt_y            0.0192571531    -0.0243063369    -0.0122431704
## magnet_belt_z            0.0081966580    -0.0050266352    -0.0088444984
## roll_arm                 0.0058323669    -0.1092692536     0.0057595520
## pitch_arm                0.0045655965     0.0085868286     0.0067242984
## yaw_arm                 -0.0003540145    -0.0793515471     0.0122956454
## total_accel_arm          0.0016383032     0.0415940123    -0.0033487201
## gyros_arm_x             -0.0169299516    -0.0019980526     0.0026870961
## gyros_arm_y              0.0116547949     0.0006968087     0.0005772699
## gyros_arm_z             -0.0250029164     0.0737587785     0.0046810084
## accel_arm_x              0.0325900293    -0.0508901908    -0.0149792104
## accel_arm_y              0.0125454027    -0.0660778967    -0.0052914462
## accel_arm_z              0.0175831320    -0.0398407132    -0.0004875015
## magnet_arm_x             0.0086841005    -0.0130107361    -0.0104230147
## magnet_arm_y             0.0179764191    -0.0063638530    -0.0048750596
## magnet_arm_z            -0.0068951541    -0.0189435879     0.0109198090
## roll_dumbbell           -0.0030567638     0.0312552328     0.0004172684
## pitch_dumbbell           0.0046753651    -0.1003045234    -0.0048699574
## yaw_dumbbell            -0.0125749440    -0.0594443735     0.0152800658
## total_accel_dumbbell     0.0032018296     0.0813129350     0.0111543948
## gyros_dumbbell_x         1.0000000000    -0.7559954775    -0.9873456420
## gyros_dumbbell_y        -0.7559954775     1.0000000000     0.7141769173
## gyros_dumbbell_z        -0.9873456420     0.7141769173     1.0000000000
## accel_dumbbell_x         0.0232834488    -0.1208745622    -0.0305144957
## accel_dumbbell_y        -0.0015907897     0.0753514015     0.0025135350
## accel_dumbbell_z        -0.0490858288    -0.0564842936     0.0483532262
## magnet_dumbbell_x        0.0478495056    -0.0734584029    -0.0101075960
## magnet_dumbbell_y       -0.0475329006     0.0934884805     0.0038255016
## magnet_dumbbell_z       -0.0042466155    -0.0826567286    -0.0029540666
## roll_forearm             0.0498837278    -0.0513065531    -0.0243143674
## pitch_forearm            0.0040439345    -0.0051747623    -0.0003603943
## yaw_forearm              0.0276831612    -0.0256867151    -0.0187118206
## total_accel_forearm     -0.0694503571     0.0168241489     0.0692999003
## gyros_forearm_x          0.3126771993    -0.2452344476    -0.3053998541
## gyros_forearm_y         -0.7940186416     0.6130265302     0.7971002225
## gyros_forearm_z         -0.9461661960     0.6887690099     0.9576090027
## accel_forearm_x          0.0442237482    -0.0303228857    -0.0314614573
## accel_forearm_y         -0.0292087636     0.0198598289     0.0315610105
## accel_forearm_z         -0.0157150282    -0.0496243124    -0.0044969725
## magnet_forearm_x        -0.0043867374    -0.0015726751     0.0076959394
## magnet_forearm_y        -0.0017368000     0.0477817592    -0.0089407927
## magnet_forearm_z         0.0385538021     0.0801484707    -0.0239902577
##                      accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## roll_belt                 0.215038108     -0.272598293      0.107457301
## pitch_belt                0.167738007     -0.443817856      0.537152776
## yaw_belt                  0.044369648      0.043705281     -0.227479986
## total_accel_belt          0.186151009     -0.271607707      0.115478536
## gyros_belt_x             -0.126385570      0.295689475     -0.348709159
## gyros_belt_y              0.013374033      0.041347100     -0.184073100
## gyros_belt_z             -0.223679665      0.201032284     -0.246847729
## accel_belt_x             -0.140863837      0.416918306     -0.513229476
## accel_belt_y              0.275158617     -0.412123875      0.281775482
## accel_belt_z             -0.266657929      0.330890612     -0.174050123
## magnet_belt_x            -0.195736066      0.407541406     -0.500677173
## magnet_belt_y            -0.370576613      0.411453948     -0.409987989
## magnet_belt_z            -0.236705650      0.317319014     -0.347194297
## roll_arm                  0.252345802     -0.169710700      0.340153486
## pitch_arm                -0.178373309      0.073128576     -0.129997042
## yaw_arm                   0.114512023     -0.084364823      0.206821358
## total_accel_arm          -0.118251954      0.109810917     -0.087054468
## gyros_arm_x              -0.019964891      0.021759276     -0.044671686
## gyros_arm_y              -0.032876509      0.017172085      0.028088343
## gyros_arm_z               0.064471991     -0.019771883     -0.099609768
## accel_arm_x               0.233017950     -0.206162118      0.296784817
## accel_arm_y              -0.249529280      0.235492391     -0.101063783
## accel_arm_z               0.129934504     -0.208969715      0.170564194
## magnet_arm_x              0.120624996      0.016498886      0.057216196
## magnet_arm_y             -0.207682886      0.083216949     -0.163386345
## magnet_arm_z             -0.074127427     -0.034340837     -0.022564024
## roll_dumbbell            -0.070679767      0.718389362     -0.321844816
## pitch_dumbbell            0.810284410     -0.159022846      0.574466784
## yaw_dumbbell              0.541576965     -0.540491702      0.847610705
## total_accel_dumbbell     -0.675797236      0.773195400     -0.625806547
## gyros_dumbbell_x          0.023283449     -0.001590790     -0.049085829
## gyros_dumbbell_y         -0.120874562      0.075351401     -0.056484294
## gyros_dumbbell_z         -0.030514496      0.002513535      0.048353226
## accel_dumbbell_x          1.000000000     -0.409538499      0.682974658
## accel_dumbbell_y         -0.409538499      1.000000000     -0.661179739
## accel_dumbbell_z          0.682974658     -0.661179739      1.000000000
## magnet_dumbbell_x         0.430501790     -0.269346049      0.535342527
## magnet_dumbbell_y        -0.125650659      0.491373136     -0.381085023
## magnet_dumbbell_z         0.051605181      0.223067726      0.031060967
## roll_forearm             -0.159347787      0.143987414     -0.113613691
## pitch_forearm             0.369580225     -0.233086294      0.271334184
## yaw_forearm              -0.186582759      0.173129991     -0.112861454
## total_accel_forearm      -0.007084404      0.155996788     -0.119683851
## gyros_forearm_x          -0.058426551      0.174946525     -0.284720964
## gyros_forearm_y          -0.028081950      0.043862507     -0.002992202
## gyros_forearm_z          -0.037003055      0.045927741      0.010214528
## accel_forearm_x          -0.204865558      0.166351182     -0.116868970
## accel_forearm_y          -0.108380217      0.164927907     -0.172052562
## accel_forearm_z           0.278148258     -0.165064727      0.236653441
## magnet_forearm_x         -0.027040284      0.072681463     -0.075282020
## magnet_forearm_y         -0.179647988      0.042845567     -0.093774492
## magnet_forearm_z         -0.303611357      0.279888377     -0.390346992
##                      magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## roll_belt                 0.3040100799     -0.2869710361      -0.501516005
## pitch_belt                0.4636106243     -0.3557730617      -0.254556348
## yaw_belt                 -0.0360777542     -0.0364661974      -0.217126645
## total_accel_belt          0.3559000791     -0.3368313652      -0.516045369
## gyros_belt_x             -0.7415554031      0.7759678215       0.224330678
## gyros_belt_y             -0.3478205754      0.3741062513      -0.169567215
## gyros_belt_z             -0.4961743700      0.4261086535       0.263816102
## accel_belt_x             -0.4777000208      0.3792110316       0.246940908
## accel_belt_y              0.4641181571     -0.4237906458      -0.583427996
## accel_belt_z             -0.3392946968      0.3182552010       0.506461923
## magnet_belt_x            -0.3533850063      0.2471316907       0.222130258
## magnet_belt_y            -0.2442457428      0.2311260733      -0.183522291
## magnet_belt_z            -0.2858776171      0.2530480505      -0.299468735
## roll_arm                  0.2154071732     -0.2603043658       0.358315717
## pitch_arm                -0.0017227702     -0.0555335002       0.011149811
## yaw_arm                   0.1389207359     -0.1567696034       0.244345250
## total_accel_arm          -0.0462264513      0.0357572699       0.134259442
## gyros_arm_x              -0.0750459897      0.0661254039       0.004438266
## gyros_arm_y               0.0299377248     -0.0278510504       0.098846406
## gyros_arm_z              -0.1814063627      0.2139526856      -0.225933121
## accel_arm_x               0.4818774509     -0.4197009797      -0.101566737
## accel_arm_y              -0.0801095080      0.0390877501       0.432766770
## accel_arm_z               0.3143362684     -0.3303263295      -0.265339030
## magnet_arm_x              0.0356330564      0.0494464237       0.009248756
## magnet_arm_y              0.0750508861     -0.1461663552      -0.129930844
## magnet_arm_z             -0.0294107066     -0.0271435243      -0.088707005
## roll_dumbbell            -0.1586164885      0.4112970236       0.126769700
## pitch_dumbbell            0.4909896643     -0.1196134541       0.068137453
## yaw_dumbbell              0.5774756534     -0.4378111349      -0.115061320
## total_accel_dumbbell     -0.1045695845      0.1607941891       0.149550181
## gyros_dumbbell_x          0.0478495056     -0.0475329006      -0.004246616
## gyros_dumbbell_y         -0.0734584029      0.0934884805      -0.082656729
## gyros_dumbbell_z         -0.0101075960      0.0038255016      -0.002954067
## accel_dumbbell_x          0.4305017904     -0.1256506587       0.051605181
## accel_dumbbell_y         -0.2693460488      0.4913731364       0.223067726
## accel_dumbbell_z          0.5353425268     -0.3810850231       0.031060967
## magnet_dumbbell_x         1.0000000000     -0.7630028954      -0.165166190
## magnet_dumbbell_y        -0.7630028954      1.0000000000       0.254316596
## magnet_dumbbell_z        -0.1651661903      0.2543165955       1.000000000
## roll_forearm              0.1709006433     -0.1711586943      -0.053227354
## pitch_forearm             0.2771505846     -0.1516214728      -0.175606656
## yaw_forearm               0.1057103217     -0.1735919787       0.155274737
## total_accel_forearm      -0.0274569659      0.0421215462       0.099394332
## gyros_forearm_x          -0.1005238513      0.0175333635      -0.089737837
## gyros_forearm_y          -0.0006539331      0.0124994238      -0.017053165
## gyros_forearm_z          -0.0015524297     -0.0001807286       0.035629353
## accel_forearm_x           0.0719314565     -0.1563415080       0.258991262
## accel_forearm_y          -0.0062329014     -0.1174402377       0.142727552
## accel_forearm_z          -0.1872163832      0.2647167400       0.602202382
## magnet_forearm_x         -0.1041223311      0.0833020965       0.247056435
## magnet_forearm_y         -0.0479169361     -0.0292785945      -0.039596075
## magnet_forearm_z          0.0259227415     -0.0394139331      -0.579061686
##                      roll_forearm pitch_forearm  yaw_forearm
## roll_belt            -0.149274616  0.1760184202 -0.269135128
## pitch_belt            0.146067842  0.2553901472  0.046977325
## yaw_belt             -0.181037901 -0.0311557633 -0.196304276
## total_accel_belt     -0.112485861  0.1859872670 -0.239711765
## gyros_belt_x         -0.226174601 -0.1570135618 -0.184873580
## gyros_belt_y         -0.219899636 -0.0161283603 -0.277436736
## gyros_belt_z          0.002190932 -0.1583712556  0.086020272
## accel_belt_x         -0.170758887 -0.2367426302 -0.078204696
## accel_belt_y         -0.104592573  0.2560568932 -0.257095914
## accel_belt_z          0.156155384 -0.2049517772  0.278857319
## magnet_belt_x        -0.158309710 -0.2438508176 -0.022633076
## magnet_belt_y         0.156488959 -0.1280541158  0.082469667
## magnet_belt_z         0.050781715 -0.0919236986  0.011397044
## roll_arm              0.074416696  0.1672488044  0.234868082
## pitch_arm             0.022917953 -0.1526173263  0.081323519
## yaw_arm               0.061856046  0.0630956300  0.190557501
## total_accel_arm       0.005111745 -0.1832025322  0.079662326
## gyros_arm_x          -0.017425233 -0.0758262899  0.051468532
## gyros_arm_y           0.035760266  0.0366024300  0.011337904
## gyros_arm_z          -0.197909266  0.0233618705 -0.248064785
## accel_arm_x           0.057526725  0.3751407442 -0.035161223
## accel_arm_y           0.194030268 -0.2074730549  0.323927259
## accel_arm_z           0.011741916  0.1489578076 -0.041658360
## magnet_arm_x         -0.020374750  0.3396281167 -0.135619629
## magnet_arm_y          0.095451213 -0.2697038747  0.152781715
## magnet_arm_z          0.014032625 -0.1322199075  0.051896242
## roll_dumbbell        -0.021264482 -0.1075936493 -0.001198079
## pitch_dumbbell       -0.074561975  0.2932128222 -0.101232618
## yaw_dumbbell          0.012820513  0.2835506665 -0.064587125
## total_accel_dumbbell  0.234068907 -0.3384650754  0.239927314
## gyros_dumbbell_x      0.049883728  0.0040439345  0.027683161
## gyros_dumbbell_y     -0.051306553 -0.0051747623 -0.025686715
## gyros_dumbbell_z     -0.024314367 -0.0003603943 -0.018711821
## accel_dumbbell_x     -0.159347787  0.3695802249 -0.186582759
## accel_dumbbell_y      0.143987414 -0.2330862936  0.173129991
## accel_dumbbell_z     -0.113613691  0.2713341837 -0.112861454
## magnet_dumbbell_x     0.170900643  0.2771505846  0.105710322
## magnet_dumbbell_y    -0.171158694 -0.1516214728 -0.173591979
## magnet_dumbbell_z    -0.053227354 -0.1756066558  0.155274737
## roll_forearm          1.000000000 -0.0513399844  0.345314013
## pitch_forearm        -0.051339984  1.0000000000 -0.220208265
## yaw_forearm           0.345314013 -0.2202082650  1.000000000
## total_accel_forearm  -0.086425370 -0.1799578173  0.190910803
## gyros_forearm_x      -0.055527715 -0.1316048817 -0.015007690
## gyros_forearm_y       0.009727854 -0.0080242224  0.004658662
## gyros_forearm_z      -0.003269748 -0.0205052305  0.014351958
## accel_forearm_x       0.291900751 -0.4697575895  0.290649050
## accel_forearm_y       0.247306788 -0.2566720252  0.399196911
## accel_forearm_z      -0.281761297 -0.0428630831 -0.133818494
## magnet_forearm_x      0.022852753 -0.4338979897 -0.074864575
## magnet_forearm_y      0.163642643 -0.2706575356  0.385363434
## magnet_forearm_z      0.407250511 -0.0133096025  0.062948799
##                      total_accel_forearm gyros_forearm_x gyros_forearm_y
## roll_belt                    0.077820127     0.377596639    0.0341411374
## pitch_belt                  -0.330906214    -0.529386526   -0.0578414424
## yaw_belt                     0.238304889     0.561597909    0.0607783799
## total_accel_belt             0.046197621     0.347139709    0.0245940338
## gyros_belt_x                 0.065101377     0.129853571    0.0218542266
## gyros_belt_y                 0.078356846     0.310077701   -0.0191473846
## gyros_belt_z                 0.014415010    -0.101310488   -0.0220192267
## accel_belt_x                 0.309017295     0.521577564    0.0563072450
## accel_belt_y                -0.025131778     0.221841371    0.0144085967
## accel_belt_z                -0.060035285    -0.349933976   -0.0283514044
## magnet_belt_x                0.259803034     0.537026032    0.0557269682
## magnet_belt_y                0.006138411     0.009722665    0.0335950469
## magnet_belt_z                0.062649577     0.115510013    0.0371874431
## roll_arm                    -0.029409455    -0.170529379   -0.0245022073
## pitch_arm                    0.103965645     0.110962223    0.0214175384
## yaw_arm                     -0.113437891    -0.105711834   -0.0263076641
## total_accel_arm             -0.076805225    -0.169395176    0.0272881204
## gyros_arm_x                  0.047664681     0.040118912   -0.0190502065
## gyros_arm_y                 -0.059644924    -0.115151559   -0.0052749228
## gyros_arm_z                  0.077494219     0.259459021    0.0126257995
## accel_arm_x                  0.002445430    -0.059480527    0.0028476535
## accel_arm_y                 -0.061887458    -0.270871668   -0.0518852903
## accel_arm_z                  0.051916458     0.166106807   -0.0245456527
## magnet_arm_x                 0.040449887    -0.015024442    0.0135684965
## magnet_arm_y                -0.018211195     0.086256805   -0.0307675229
## magnet_arm_z                 0.008255227     0.089528538   -0.0214788659
## roll_dumbbell                0.181086546     0.169833644    0.0041289650
## pitch_dumbbell               0.006826758    -0.094547527   -0.0300114076
## yaw_dumbbell                -0.195302359    -0.311073610   -0.0320061509
## total_accel_dumbbell         0.119339465     0.130724416    0.0691767137
## gyros_dumbbell_x            -0.069450357     0.312677199   -0.7940186416
## gyros_dumbbell_y             0.016824149    -0.245234448    0.6130265302
## gyros_dumbbell_z             0.069299900    -0.305399854    0.7971002225
## accel_dumbbell_x            -0.007084404    -0.058426551   -0.0280819495
## accel_dumbbell_y             0.155996788     0.174946525    0.0438625065
## accel_dumbbell_z            -0.119683851    -0.284720964   -0.0029922016
## magnet_dumbbell_x           -0.027456966    -0.100523851   -0.0006539331
## magnet_dumbbell_y            0.042121546     0.017533364    0.0124994238
## magnet_dumbbell_z            0.099394332    -0.089737837   -0.0170531649
## roll_forearm                -0.086425370    -0.055527715    0.0097278541
## pitch_forearm               -0.179957817    -0.131604882   -0.0080242224
## yaw_forearm                  0.190910803    -0.015007690    0.0046586622
## total_accel_forearm          1.000000000     0.215594802    0.0620864920
## gyros_forearm_x              0.215594802     1.000000000   -0.4700723809
## gyros_forearm_y              0.062086492    -0.470072381    1.0000000000
## gyros_forearm_z              0.097954549    -0.312888417    0.8855175370
## accel_forearm_x             -0.209524121    -0.122465059   -0.0442956295
## accel_forearm_y              0.293299069     0.285322427    0.0370236848
## accel_forearm_z              0.007856124     0.008627819   -0.0116214251
## magnet_forearm_x            -0.259988841    -0.032566501    0.0026743778
## magnet_forearm_y             0.134682734     0.094865655    0.0042644379
## magnet_forearm_z            -0.015381916     0.175326244    0.0580903945
##                      gyros_forearm_z accel_forearm_x accel_forearm_y
## roll_belt               0.0153708327    -0.494602490     0.030049348
## pitch_belt             -0.0752345765     0.129248588    -0.366305475
## yaw_belt                0.0540690837    -0.399430467     0.240357900
## total_accel_belt        0.0064209058    -0.456884579     0.019932474
## gyros_belt_x            0.0141857615    -0.219437198    -0.041354639
## gyros_belt_y            0.0096669551    -0.391729159    -0.021491785
## gyros_belt_z            0.0004446853     0.139440370    -0.006267761
## accel_belt_x            0.0746078394    -0.172389454     0.345141203
## accel_belt_y           -0.0066761462    -0.451241079    -0.078643140
## accel_belt_z           -0.0120702584     0.490904351    -0.020243300
## magnet_belt_x           0.0725468379    -0.174166567     0.332599591
## magnet_belt_y          -0.0091817992     0.070903933    -0.038625920
## magnet_belt_z          -0.0136805062    -0.085317236    -0.007287379
## roll_arm                0.0131447699     0.226344597     0.064847841
## pitch_arm               0.0287062501     0.021328742     0.093315025
## yaw_arm                 0.0046760966     0.160033086     0.028977388
## total_accel_arm         0.0165692356     0.245074915    -0.045252363
## gyros_arm_x             0.0009575252    -0.042277119     0.112560820
## gyros_arm_y            -0.0064623833     0.142609905    -0.117158741
## gyros_arm_z             0.0111623090    -0.427429735    -0.003262478
## accel_arm_x            -0.0007257841    -0.142293345    -0.044857837
## accel_arm_y            -0.0189661095     0.522565065     0.036536955
## accel_arm_z            -0.0177676621    -0.158377477     0.051882250
## magnet_arm_x            0.0026158186    -0.231677899    -0.078722545
## magnet_arm_y           -0.0193419093     0.201712138     0.075565398
## magnet_arm_z           -0.0128106443     0.041525036     0.069971115
## roll_dumbbell           0.0203444762     0.002820354     0.084181421
## pitch_dumbbell         -0.0176407116    -0.082353669    -0.099861020
## yaw_dumbbell           -0.0320231969    -0.015929397    -0.199101946
## total_accel_dumbbell    0.0614776188     0.266504436     0.183372943
## gyros_dumbbell_x       -0.9461661960     0.044223748    -0.029208764
## gyros_dumbbell_y        0.6887690099    -0.030322886     0.019859829
## gyros_dumbbell_z        0.9576090027    -0.031461457     0.031561011
## accel_dumbbell_x       -0.0370030548    -0.204865558    -0.108380217
## accel_dumbbell_y        0.0459277408     0.166351182     0.164927907
## accel_dumbbell_z        0.0102145282    -0.116868970    -0.172052562
## magnet_dumbbell_x      -0.0015524297     0.071931456    -0.006232901
## magnet_dumbbell_y      -0.0001807286    -0.156341508    -0.117440238
## magnet_dumbbell_z       0.0356293530     0.258991262     0.142727552
## roll_forearm           -0.0032697480     0.291900751     0.247306788
## pitch_forearm          -0.0205052305    -0.469757589    -0.256672025
## yaw_forearm             0.0143519583     0.290649050     0.399196911
## total_accel_forearm     0.0979545491    -0.209524121     0.293299069
## gyros_forearm_x        -0.3128884170    -0.122465059     0.285322427
## gyros_forearm_y         0.8855175370    -0.044295630     0.037023685
## gyros_forearm_z         1.0000000000    -0.028167862     0.082642441
## accel_forearm_x        -0.0281678624     1.000000000     0.308042102
## accel_forearm_y         0.0826424412     0.308042102     1.000000000
## accel_forearm_z         0.0258684632    -0.204162656     0.046360348
## magnet_forearm_x        0.0034817606     0.679500025    -0.132809121
## magnet_forearm_y        0.0121822707     0.211580371     0.772035803
## magnet_forearm_z        0.0013929185     0.065427600     0.284754075
##                      accel_forearm_z magnet_forearm_x magnet_forearm_y
## roll_belt                0.083836359     -0.202898875      0.025282893
## pitch_belt              -0.229668266     -0.066197442     -0.018887667
## yaw_belt                 0.172420735     -0.101369131      0.043467517
## total_accel_belt         0.046196173     -0.197231811      0.036895492
## gyros_belt_x             0.347881453      0.044173474      0.018245599
## gyros_belt_y             0.262334184     -0.042777655      0.004831068
## gyros_belt_z             0.093520822      0.124458017      0.003889148
## accel_belt_x             0.290135322      0.045179445      0.034688753
## accel_belt_y             0.015658618     -0.220360504      0.017309147
## accel_belt_z            -0.090077173      0.210340125     -0.023466119
## magnet_belt_x            0.298215608      0.021975324      0.075950970
## magnet_belt_y           -0.355316876      0.015668218      0.004548481
## magnet_belt_z           -0.362546062     -0.089072663      0.042000279
## roll_arm                 0.075433324      0.056887289     -0.094966389
## pitch_arm               -0.002018454      0.012677203      0.055419612
## yaw_arm                  0.088449440      0.050836163     -0.020473181
## total_accel_arm         -0.057175544      0.157518570     -0.006059733
## gyros_arm_x              0.061344431     -0.065367852      0.166191813
## gyros_arm_y             -0.076065930      0.096079237     -0.152218663
## gyros_arm_z              0.203143815     -0.135693644      0.057206634
## accel_arm_x             -0.070019453     -0.183599605     -0.137847274
## accel_arm_y             -0.104493211      0.195745565      0.028325771
## accel_arm_z             -0.070075482     -0.133836614      0.038149641
## magnet_arm_x             0.032248529     -0.152978048     -0.169736404
## magnet_arm_y            -0.174000458      0.091921730      0.150831945
## magnet_arm_z            -0.030933766      0.020327580      0.132649904
## roll_dumbbell           -0.066822480      0.012992869     -0.031573283
## pitch_dumbbell           0.125451113     -0.026269098     -0.154951515
## yaw_dumbbell            -0.017199308     -0.094515017     -0.066823967
## total_accel_dumbbell    -0.250708576      0.081002929      0.093078025
## gyros_dumbbell_x        -0.015715028     -0.004386737     -0.001736800
## gyros_dumbbell_y        -0.049624312     -0.001572675      0.047781759
## gyros_dumbbell_z        -0.004496972      0.007695939     -0.008940793
## accel_dumbbell_x         0.278148258     -0.027040284     -0.179647988
## accel_dumbbell_y        -0.165064727      0.072681463      0.042845567
## accel_dumbbell_z         0.236653441     -0.075282020     -0.093774492
## magnet_dumbbell_x       -0.187216383     -0.104122331     -0.047916936
## magnet_dumbbell_y        0.264716740      0.083302096     -0.029278594
## magnet_dumbbell_z        0.602202382      0.247056435     -0.039596075
## roll_forearm            -0.281761297      0.022852753      0.163642643
## pitch_forearm           -0.042863083     -0.433897990     -0.270657536
## yaw_forearm             -0.133818494     -0.074864575      0.385363434
## total_accel_forearm      0.007856124     -0.259988841      0.134682734
## gyros_forearm_x          0.008627819     -0.032566501      0.094865655
## gyros_forearm_y         -0.011621425      0.002674378      0.004264438
## gyros_forearm_z          0.025868463      0.003481761      0.012182271
## accel_forearm_x         -0.204162656      0.679500025      0.211580371
## accel_forearm_y          0.046360348     -0.132809121      0.772035803
## accel_forearm_z          1.000000000      0.068486128      0.035273836
## magnet_forearm_x         0.068486128      1.000000000     -0.297315614
## magnet_forearm_y         0.035273836     -0.297315614      1.000000000
## magnet_forearm_z        -0.543903130     -0.131189632      0.242040604
##                      magnet_forearm_z
## roll_belt                 0.272236474
## pitch_belt               -0.071160028
## yaw_belt                  0.231730193
## total_accel_belt          0.295290478
## gyros_belt_x             -0.112290313
## gyros_belt_y              0.121738819
## gyros_belt_z             -0.113352808
## accel_belt_x              0.046962441
## accel_belt_y              0.245927666
## accel_belt_z             -0.240065982
## magnet_belt_x             0.047325020
## magnet_belt_y             0.379616801
## magnet_belt_z             0.393219256
## roll_arm                 -0.337649202
## pitch_arm                -0.026043777
## yaw_arm                  -0.235142201
## total_accel_arm          -0.097250422
## gyros_arm_x               0.013665944
## gyros_arm_y              -0.098164769
## gyros_arm_z               0.126892516
## accel_arm_x               0.044589383
## accel_arm_y              -0.222749835
## accel_arm_z               0.102126037
## magnet_arm_x              0.050102420
## magnet_arm_y              0.119937422
## magnet_arm_z              0.005862917
## roll_dumbbell             0.151485579
## pitch_dumbbell           -0.216918057
## yaw_dumbbell             -0.196418399
## total_accel_dumbbell      0.305132715
## gyros_dumbbell_x          0.038553802
## gyros_dumbbell_y          0.080148471
## gyros_dumbbell_z         -0.023990258
## accel_dumbbell_x         -0.303611357
## accel_dumbbell_y          0.279888377
## accel_dumbbell_z         -0.390346992
## magnet_dumbbell_x         0.025922742
## magnet_dumbbell_y        -0.039413933
## magnet_dumbbell_z        -0.579061686
## roll_forearm              0.407250511
## pitch_forearm            -0.013309602
## yaw_forearm               0.062948799
## total_accel_forearm      -0.015381916
## gyros_forearm_x           0.175326244
## gyros_forearm_y           0.058090394
## gyros_forearm_z           0.001392919
## accel_forearm_x           0.065427600
## accel_forearm_y           0.284754075
## accel_forearm_z          -0.543903130
## magnet_forearm_x         -0.131189632
## magnet_forearm_y          0.242040604
## magnet_forearm_z          1.000000000
```

lets get a better look at the frequency of levels in the subtraining data set. level A in the red is the more frequent occurance with over 4000 and level D in the yellow has the lease with about 2500. 


```r
plot2 <- plot(subdatatraining$classe, col=rainbow(20), main="Levels of classe within the subdatatraining dataset", xlab="classe levels", ylab="Frequency")
```

![](Predictions_files/figure-html/unnamed-chunk-9-1.png)

```r
plot2
```

```
##      [,1]
## [1,]  0.7
## [2,]  1.9
## [3,]  3.1
## [4,]  4.3
## [5,]  5.5
```

## Prediction Models
A prediction model using a decision tree. 


```r
model1 <- rpart(classe ~ ., data=subdatatraining, method="class")
prediction1 <- predict(model1, subdatatesting, type = "class")
plot3 <- rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0)
```

![](Predictions_files/figure-html/unnamed-chunk-10-1.png)

```r
plot3
```

```
## $obj
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##     1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##       2) roll_belt< 130.5 10783 7444 A (0.31 0.21 0.19 0.18 0.11)  
##         4) pitch_forearm< -33.95 948    7 A (0.99 0.0074 0 0 0) *
##         5) pitch_forearm>=-33.95 9835 7437 A (0.24 0.23 0.21 0.2 0.12)  
##          10) magnet_dumbbell_y< 439.5 8332 5979 A (0.28 0.18 0.24 0.19 0.11)  
##            20) roll_forearm< 124.5 5236 3112 A (0.41 0.18 0.18 0.17 0.06)  
##              40) magnet_dumbbell_z< -24.5 1844  633 A (0.66 0.21 0.022 0.079 0.03) *
##              41) magnet_dumbbell_z>=-24.5 3392 2465 C (0.27 0.17 0.27 0.21 0.076)  
##                82) yaw_belt>=168.5 472   82 A (0.83 0.089 0.0021 0.081 0.0021) *
##                83) yaw_belt< 168.5 2920 1994 C (0.18 0.18 0.32 0.24 0.088)  
##                 166) pitch_belt< -43.15 267   34 B (0.015 0.87 0.06 0.03 0.022) *
##                 167) pitch_belt>=-43.15 2653 1743 C (0.2 0.11 0.34 0.26 0.095)  
##                   334) accel_dumbbell_y>=-39.5 2255 1589 D (0.23 0.12 0.25 0.3 0.11)  
##                     668) roll_belt>=125.5 532  198 C (0.33 0.032 0.63 0.011 0.0038)  
##                      1336) magnet_belt_z< -326 152    3 A (0.98 0 0.0066 0 0.013) *
##                      1337) magnet_belt_z>=-326 380   47 C (0.063 0.045 0.88 0.016 0) *
##                     669) roll_belt< 125.5 1723 1063 D (0.2 0.15 0.13 0.38 0.14)  
##                      1338) pitch_belt>=1.3 1062  806 A (0.24 0.22 0.13 0.21 0.2)  
##                        2676) accel_dumbbell_z< 30.5 694  451 A (0.35 0.15 0.2 0.26 0.042)  
##                          5352) yaw_forearm>=-94.75 503  260 A (0.48 0.19 0.22 0.064 0.044)  
##                           10704) magnet_forearm_z>=-69.5 313   81 A (0.74 0.12 0.032 0.08 0.026) *
##                           10705) magnet_forearm_z< -69.5 190   90 C (0.058 0.31 0.53 0.037 0.074) *
##                          5353) yaw_forearm< -94.75 191   43 D (0 0.031 0.16 0.77 0.037) *
##                        2677) accel_dumbbell_z>=30.5 368  185 E (0.035 0.35 0.0054 0.11 0.5) *
##                      1339) pitch_belt< 1.3 661  222 D (0.13 0.036 0.13 0.66 0.035)  
##                        2678) yaw_arm< -111 86    0 A (1 0 0 0 0) *
##                        2679) yaw_arm>=-111 575  136 D (0.0017 0.042 0.15 0.76 0.04) *
##                   335) accel_dumbbell_y< -39.5 398   52 C (0.0075 0.05 0.87 0.038 0.035) *
##            21) roll_forearm>=124.5 3096 2076 C (0.074 0.18 0.33 0.23 0.19)  
##              42) accel_forearm_x>=-107.5 2175 1360 C (0.084 0.21 0.37 0.1 0.23)  
##                84) magnet_dumbbell_z>=283.5 548  386 A (0.3 0.29 0.016 0.14 0.26)  
##                 168) accel_dumbbell_z< 28.5 184   30 A (0.84 0.082 0.038 0.043 0) *
##                 169) accel_dumbbell_z>=28.5 364  221 B (0.022 0.39 0.0055 0.2 0.38)  
##                   338) yaw_belt< -88.15 165   28 B (0.03 0.83 0.012 0.024 0.1) *
##                   339) yaw_belt>=-88.15 199   76 E (0.015 0.03 0 0.34 0.62) *
##                85) magnet_dumbbell_z< 283.5 1627  821 C (0.012 0.19 0.5 0.089 0.22)  
##                 170) gyros_belt_z< 0.06 1473  669 C (0.014 0.2 0.55 0.073 0.16) *
##                 171) gyros_belt_z>=0.06 154   46 E (0 0.039 0.013 0.25 0.7) *
##              43) accel_forearm_x< -107.5 921  421 D (0.051 0.088 0.22 0.54 0.096)  
##                86) magnet_arm_y>=291 283  117 C (0.035 0.14 0.59 0.17 0.06) *
##                87) magnet_arm_y< 291 638  187 D (0.058 0.063 0.061 0.71 0.11) *
##          11) magnet_dumbbell_y>=439.5 1503  734 B (0.03 0.51 0.045 0.22 0.19)  
##            22) total_accel_dumbbell>=5.5 1073  373 B (0.042 0.65 0.061 0.019 0.23)  
##              44) roll_belt>=-0.575 900  200 B (0.05 0.78 0.072 0.022 0.078) *
##              45) roll_belt< -0.575 173    0 E (0 0 0 0 1) *
##            23) total_accel_dumbbell< 5.5 430  116 D (0 0.16 0.0047 0.73 0.1) *
##       3) roll_belt>=130.5 993    9 E (0.0091 0 0 0 0.99) *
## 
## $snipped.nodes
## NULL
## 
## $xlim
## [1] 0 1
## 
## $ylim
## [1] 0 1
## 
## $x
##  [1] 0.61843916 0.56612869 0.49254438 0.63971300 0.35339033 0.09428547
##  [7] 0.04433190 0.14423905 0.09422368 0.19425442 0.14362112 0.24488773
## [13] 0.18807509 0.06946913 0.02451709 0.11442117 0.30668105 0.21430461
## [19] 0.16440784 0.11496060 0.07000856 0.15991264 0.21385509 0.26420137
## [25] 0.39905750 0.35410545 0.44400954 0.30170037 0.61249518 0.49224847
## [31] 0.39560158 0.34615434 0.44504883 0.40009679 0.49000087 0.58889536
## [37] 0.54394332 0.63384740 0.73274189 0.68778985 0.77769393 0.92603566
## [43] 0.87658842 0.83163638 0.92154046 0.97548291 0.67074963
## 
## $y
##  [1] 0.97649288 0.90338261 0.83027233 0.83027233 0.75716206 0.68405179
##  [7] 0.61094151 0.61094151 0.53783124 0.53783124 0.46472096 0.46472096
## [13] 0.39161069 0.31850042 0.24539014 0.24539014 0.31850042 0.24539014
## [19] 0.17227987 0.09916959 0.02605932 0.02605932 0.09916959 0.17227987
## [25] 0.24539014 0.17227987 0.17227987 0.39161069 0.68405179 0.61094151
## [31] 0.53783124 0.46472096 0.46472096 0.39161069 0.39161069 0.53783124
## [37] 0.46472096 0.46472096 0.61094151 0.53783124 0.53783124 0.75716206
## [43] 0.68405179 0.61094151 0.61094151 0.68405179 0.90338261
## 
## $branch.x
##        [,1]      [,2]      [,3]      [,4]      [,5]       [,6]       [,7]
## x 0.6184392 0.5661287 0.4925444 0.6397130 0.3533903 0.09428547 0.04433190
##          NA 0.6079771 0.5514118 0.5808455 0.5824485 0.30156936 0.08429476
##          NA 0.6184392 0.5661287 0.5661287 0.6397130 0.35339033 0.09428547
##         [,8]       [,9]     [,10]     [,11]     [,12]     [,13]      [,14]
## x 0.14423905 0.09422368 0.1942544 0.1436211 0.2448877 0.1880751 0.06946913
##   0.10427619 0.13423598 0.1542421 0.1841278 0.2043811 0.2335252 0.16435390
##   0.09428547 0.14423905 0.1442391 0.1942544 0.1942544 0.2448877 0.18807509
##        [,15]      [,16]     [,17]     [,18]     [,19]     [,20]      [,21]
## x 0.02451709 0.11442117 0.3066811 0.2143046 0.1644078 0.1149606 0.07000856
##   0.06047872 0.07845954 0.2117963 0.2882058 0.2043253 0.1545184 0.10597019
##   0.06946913 0.06946913 0.1880751 0.3066811 0.2143046 0.1644078 0.11496060
##       [,22]     [,23]     [,24]     [,25]     [,26]     [,27]     [,28]
## x 0.1599126 0.2138551 0.2642014 0.3990575 0.3541055 0.4440095 0.3017004
##   0.1239510 0.1742973 0.2242840 0.3251563 0.3900671 0.4080479 0.2562503
##   0.1149606 0.1644078 0.2143046 0.3066811 0.3990575 0.3990575 0.2448877
##       [,29]     [,30]     [,31]     [,32]     [,33]     [,34]     [,35]
## x 0.6124952 0.4922485 0.3956016 0.3461543 0.4450488 0.4000968 0.4900009
##   0.4052113 0.5884458 0.4729191 0.3857121 0.4054910 0.4360584 0.4540392
##   0.3533903 0.6124952 0.4922485 0.3956016 0.3956016 0.4450488 0.4450488
##       [,36]     [,37]     [,38]     [,39]     [,40]     [,41]     [,42]
## x 0.5888954 0.5439433 0.6338474 0.7327419 0.6877898 0.7776939 0.9260357
##   0.5115778 0.5799049 0.5978858 0.6365445 0.7237515 0.7417323 0.6969775
##   0.4922485 0.5888954 0.5888954 0.6124952 0.7327419 0.7327419 0.6397130
##       [,43]     [,44]     [,45]     [,46]     [,47]
## x 0.8765884 0.8316364 0.9215405 0.9754829 0.6707496
##   0.9161462 0.8675980 0.8855788 0.9359251 0.6289013
##   0.9260357 0.8765884 0.8765884 0.9260357 0.6184392
## 
## $branch.y
##       [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]
## y 0.978839 0.9057287 0.8718239 0.8326184 0.7595082 0.6863979 0.6524930
##         NA 0.9764929 0.9033826 0.9033826 0.8302723 0.7571621 0.6840518
##         NA 0.9764929 0.9033826 0.9033826 0.8302723 0.7571621 0.6840518
##        [,8]      [,9]     [,10]     [,11]     [,12]     [,13]     [,14]
## y 0.6132876 0.5793828 0.5401773 0.5062725 0.4670671 0.3939568 0.3208465
##   0.6840518 0.6109415 0.6109415 0.5378312 0.5378312 0.4647210 0.3916107
##   0.6840518 0.6109415 0.6109415 0.5378312 0.5378312 0.4647210 0.3916107
##       [,15]     [,16]     [,17]     [,18]     [,19]     [,20]      [,21]
## y 0.2869417 0.2869417 0.3208465 0.2477362 0.1746260 0.1015157 0.06761084
##   0.3185004 0.3185004 0.3916107 0.3185004 0.2453901 0.1722799 0.09916959
##   0.3185004 0.3185004 0.3916107 0.3185004 0.2453901 0.1722799 0.09916959
##        [,22]     [,23]     [,24]     [,25]     [,26]     [,27]     [,28]
## y 0.06761084 0.1407211 0.2138314 0.2477362 0.2138314 0.2138314 0.4331622
##   0.09916959 0.1722799 0.2453901 0.3185004 0.2453901 0.2453901 0.4647210
##   0.09916959 0.1722799 0.2453901 0.3185004 0.2453901 0.2453901 0.4647210
##       [,29]     [,30]     [,31]     [,32]     [,33]     [,34]     [,35]
## y 0.6863979 0.6132876 0.5401773 0.5062725 0.4670671 0.4331622 0.4331622
##   0.7571621 0.6840518 0.6109415 0.5378312 0.5378312 0.4647210 0.4647210
##   0.7571621 0.6840518 0.6109415 0.5378312 0.5378312 0.4647210 0.4647210
##       [,36]     [,37]     [,38]     [,39]     [,40]     [,41]     [,42]
## y 0.5401773 0.5062725 0.5062725 0.6132876 0.5793828 0.5793828 0.7595082
##   0.6109415 0.5378312 0.5378312 0.6840518 0.6109415 0.6109415 0.8302723
##   0.6109415 0.5378312 0.5378312 0.6840518 0.6109415 0.6109415 0.8302723
##       [,43]     [,44]     [,45]     [,46]     [,47]
## y 0.6863979 0.6524930 0.6524930 0.7256033 0.9449341
##   0.7571621 0.6840518 0.6840518 0.7571621 0.9764929
##   0.7571621 0.6840518 0.6840518 0.7571621 0.9764929
## 
## $labs
##  [1] NA                      NA                     
##  [3] "A\n\n941 / 948\n8%"    NA                     
##  [5] NA                      NA                     
##  [7] "A\n\n1211 / 1844\n16%" NA                     
##  [9] "A\n\n390 / 472\n4%"    NA                     
## [11] "B\n\n233 / 267\n2%"    NA                     
## [13] NA                      NA                     
## [15] "A\n\n149 / 152\n1%"    "C\n\n333 / 380\n3%"   
## [17] NA                      NA                     
## [19] NA                      NA                     
## [21] "A\n\n232 / 313\n3%"    "C\n\n100 / 190\n2%"   
## [23] "D\n\n148 / 191\n2%"    "E\n\n183 / 368\n3%"   
## [25] NA                      "A\n\n86 / 86\n1%"     
## [27] "D\n\n439 / 575\n5%"    "C\n\n346 / 398\n3%"   
## [29] NA                      NA                     
## [31] NA                      "A\n\n154 / 184\n2%"   
## [33] NA                      "B\n\n137 / 165\n1%"   
## [35] "E\n\n123 / 199\n2%"    NA                     
## [37] "C\n\n804 / 1473\n13%"  "E\n\n108 / 154\n1%"   
## [39] NA                      "C\n\n166 / 283\n2%"   
## [41] "D\n\n451 / 638\n5%"    NA                     
## [43] NA                      "B\n\n700 / 900\n8%"   
## [45] "E\n\n173 / 173\n1%"    "D\n\n314 / 430\n4%"   
## [47] "E\n\n984 / 993\n8%"   
## 
## $cex
## [1] 0.375
## 
## $boxes
## $boxes$x1
##  [1]         NA         NA 0.48487392         NA         NA         NA
##  [7] 0.03666144         NA 0.08655323         NA 0.13595066         NA
## [13]         NA         NA 0.01684664 0.10675072         NA         NA
## [19]         NA         NA 0.06233810 0.15224218 0.20618463 0.25653092
## [25]         NA 0.34643500 0.43633908 0.29402991         NA         NA
## [31]         NA 0.33848388         NA 0.39242633 0.48233041         NA
## [37] 0.53627286 0.62617694         NA 0.68011939 0.77002347         NA
## [43]         NA 0.82396592 0.91387000 0.96781245 0.66307918
## 
## $boxes$y1
##  [1]        NA        NA 0.8435575        NA        NA        NA 0.6242267
##  [8]        NA 0.5511164        NA 0.4780061        NA        NA        NA
## [15] 0.2586753 0.2586753        NA        NA        NA        NA 0.0393445
## [22] 0.0393445 0.1124548 0.1855650        NA 0.1855650 0.1855650 0.4048959
## [29]        NA        NA        NA 0.4780061        NA 0.4048959 0.4048959
## [36]        NA 0.4780061 0.4780061        NA 0.5511164 0.5511164        NA
## [43]        NA 0.6242267 0.6242267 0.6973370 0.9166678
## 
## $boxes$x2
##  [1]         NA         NA 0.50021484         NA         NA         NA
##  [7] 0.05200235         NA 0.10189414         NA 0.15129158         NA
## [13]         NA         NA 0.03218755 0.12209163         NA         NA
## [19]         NA         NA 0.07767901 0.16758309 0.22152554 0.27187183
## [25]         NA 0.36177591 0.45167999 0.30937082         NA         NA
## [31]         NA 0.35382479         NA 0.40776724 0.49767132         NA
## [37] 0.55161377 0.64151785         NA 0.69546030 0.78536438         NA
## [43]         NA 0.83930683 0.92921092 0.98315336 0.67842009
## 
## $boxes$y2
##  [1]         NA         NA 0.87182385         NA         NA         NA
##  [7] 0.65249303         NA 0.57938276         NA 0.50627248         NA
## [13]         NA         NA 0.28694166 0.28694166         NA         NA
## [19]         NA         NA 0.06761084 0.06761084 0.14072111 0.21383139
## [25]         NA 0.21383139 0.21383139 0.43316221         NA         NA
## [31]         NA 0.50627248         NA 0.43316221 0.43316221         NA
## [37] 0.50627248 0.50627248         NA 0.57938276 0.57938276         NA
## [43]         NA 0.65249303 0.65249303 0.72560330 0.94493413
## 
## 
## $split.labs
## [1] ""
## 
## $split.cex
## [1] 1
## 
## $split.box
## $split.box$x1
##  [1] 0.57923461 0.51499232         NA 0.57153117 0.30566305 0.02695592
##  [7]         NA 0.10077314         NA 0.15164078         NA 0.18011499
## [13] 0.14546145 0.01407140         NA         NA 0.26065832 0.15464551
## [19] 0.11327147 0.05018786         NA         NA         NA         NA
## [25] 0.35814840         NA         NA         NA 0.55113153 0.42065755
## [31] 0.33594248         NA 0.40584428         NA         NA 0.53775899
## [37]         NA         NA 0.67649188         NA         NA 0.85870611
## [43] 0.83056568         NA         NA         NA         NA
## 
## $split.box$y1
##  [1] 0.96942630 0.89631602         NA 0.82320575 0.75009548 0.67698520
##  [7]         NA 0.60387493         NA 0.53076465         NA 0.45765438
## [13] 0.38454410 0.31143383         NA         NA 0.31143383 0.23832356
## [19] 0.16521328 0.09210301         NA         NA         NA         NA
## [25] 0.23832356         NA         NA         NA 0.67698520 0.60387493
## [31] 0.53076465         NA 0.45765438         NA         NA 0.53076465
## [37]         NA         NA 0.60387493         NA         NA 0.75009548
## [43] 0.67698520         NA         NA         NA         NA
## 
## $split.box$x2
##  [1] 0.6576437 0.6172651        NA 0.7078948 0.4011176 0.1616150        NA
##  [8] 0.1877050        NA 0.2368681        NA 0.3096605 0.2306887 0.1248669
## [15]        NA        NA 0.3527038 0.2739637 0.2155442 0.1797333        NA
## [22]        NA        NA        NA 0.4399666        NA        NA        NA
## [29] 0.6738588 0.5638394 0.4552607        NA 0.4842534        NA        NA
## [36] 0.6400317        NA        NA 0.7889919        NA        NA 0.9933652
## [43] 0.9226112        NA        NA        NA        NA
## 
## $split.box$y2
##  [1] 0.9976926 0.9245824        NA 0.8514721 0.7783618 0.7052515        NA
##  [8] 0.6321413        NA 0.5590310        NA 0.4859207 0.4128104 0.3397002
## [15]        NA        NA 0.3397002 0.2665899 0.1934796 0.1203693        NA
## [22]        NA        NA        NA 0.2665899        NA        NA        NA
## [29] 0.7052515 0.6321413 0.5590310        NA 0.4859207        NA        NA
## [36] 0.5590310        NA        NA 0.6321413        NA        NA 0.7783618
## [43] 0.7052515        NA        NA        NA        NA
```

lets test our results 


```r
confusionMatrix(prediction1, subdatatesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2064  313   34  154   44
##          B   49  708   60   23   71
##          C   70  305 1168  125  188
##          D   31   93  100  863  106
##          E   18   99    6  121 1033
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7438         
##                  95% CI : (0.734, 0.7534)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6744         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9247  0.46640   0.8538   0.6711   0.7164
## Specificity            0.9029  0.96792   0.8938   0.9497   0.9619
## Pos Pred Value         0.7911  0.77717   0.6293   0.7234   0.8089
## Neg Pred Value         0.9679  0.88320   0.9666   0.9364   0.9377
## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
## Detection Rate         0.2631  0.09024   0.1489   0.1100   0.1317
## Detection Prevalence   0.3325  0.11611   0.2366   0.1521   0.1628
## Balanced Accuracy      0.9138  0.71716   0.8738   0.8104   0.8391
```

second prediction model using random forest. 


```r
model2 <- randomForest(classe ~. , data=subdatatraining, method="class")
prediction2 <- predict(model2, subdatatesting, type = "class")
confusionMatrix(prediction2, subdatatesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    5    0    0    0
##          B    5 1509    6    0    0
##          C    1    4 1359   14    0
##          D    0    0    3 1271    5
##          E    0    0    0    1 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9925, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9941   0.9934   0.9883   0.9965
## Specificity            0.9991   0.9983   0.9971   0.9988   0.9998
## Pos Pred Value         0.9978   0.9928   0.9862   0.9937   0.9993
## Neg Pred Value         0.9989   0.9986   0.9986   0.9977   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1923   0.1732   0.1620   0.1832
## Detection Prevalence   0.2843   0.1937   0.1756   0.1630   0.1833
## Balanced Accuracy      0.9982   0.9962   0.9952   0.9936   0.9982
```

##Evaluatin of sample error, using 4 fold cross validation. 


```r
rf1<- train(x=subdatatraining[,-53],y=subdatatraining$classe,method="rf",
                trControl=trainControl(method = "cv", number = 4),
                data=subdatatraining,do.trace=F,ntree=250)
rf1
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 8832, 8832, 8831, 8833 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9865826  0.9830257  0.002520887  0.003187298
##   27    0.9873472  0.9839935  0.001826161  0.002310007
##   52    0.9798735  0.9745387  0.003989264  0.005042732
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

in sample error and out of sample error. 


```r
predictiontrainingrf1 <- predict(rf1$finalModel,newdata=subdatatraining)
insampleerrorforrf1<- 100- (mean((predictiontrainingrf1 == subdatatraining$classe)*1)*100)
insampleerrorforrf1
```

```
## [1] 0
```

```r
predictiontestingrf1 <- predict(rf1,subdatatesting)
outofsampleerrorrf1 <- 100 -(mean((predictiontestingrf1 == subdatatesting$classe)*1)*100)
outofsampleerrorrf1
```

```
## [1] 0.7519755
```

```r
predictiontestingrf1 <- predict(rf1,datatesting[,-53])
predictiontestingrf1
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
table(predictiontestingrf1)
```

```
## predictiontestingrf1
## A B C D E 
## 7 8 1 1 3
```



##Decision/Conclusion

Random Forest algorithm performed better than Decision Trees.
Accuracy for Random Forest model was 0.9944 (95% CI: (0.9925, 0.9959)) compared to 0.7438 (95% CI: (0.734, 0.7534)) for Decision Tree model. The random Forest model is choosen. The accuracy of the model is 0.9929. With an accuracy above 99% on our cross-validation data, we can expect that very few, or none, of the test samples will be missclassified.

        **Random forests are suitable when to handling a large number of inputs, especially when the interactions between variables are unknown.
        **Random forest's built in cross-validation component that gives an unbiased estimate of the forest's out-of-sample (or bag) (OOB) error rate.
        **A Random forest can handle unscaled variables and categorical variables. This is more forgiving with the cleaning of the data.
        

## Index Including Plots

corrPlot <- cor(subdatatraining[, -53])
corrplot(corrPlot, method="color")


plot(subdatatraining$classe, col=rainbow(20), main="Levels of classe within the subTraining dataset", xlab="classe levels", ylab="Frequency")
rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0)

model1 <- rpart(classe ~ ., data=subdatatraining, method="class")
prediction1 <- predict(model1, subdatatesting, type = "class")
rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0)



