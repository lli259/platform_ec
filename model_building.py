import argparse,os
import sys
import math
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def define_args(arg_parser):

    arg_parser.add_argument('--feature_folder', nargs='*', default=['features_selected'], help='Gringo input files')
    arg_parser.add_argument('--performance_folder', nargs='*', default=['performance_selected'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')
    arg_parser.add_argument('--ml_models_folder', nargs='*', default=['ml_models'], help='Gringo input files') 
    arg_parser.add_argument('--ml_hyper_folder', nargs='*', default=['ml_hyper'], help='Gringo input files') 

def checkMakeFolder(fdname):
    if not os.path.exists(fdname):
        os.makedirs(fdname)

def check_content(fdname):
    if os.listdir(fdname) == []:
        return False
    else:
        return True

def cleanFolder(fdnames):   
    ans=input('Models existed. Need to retrain models? y/n')
    if ans =='y':
        for file_in in fdnames:
            if os.path.exists(file_in):
                os.system('rm -r '+file_in+'/*')

#write to evaluation file2 
#evaluation/result2.csv
def write2eva2(algname,slv,time):
    fname='evaluation/result2.csv'
    with open (fname,'a') as f:
        cont=','.join([algname,str(slv),str(time)])
        f.write(cont+'\n')
#the objective function to minimize, tuning hyperparameters
#relative_score
#max_relative_score
#min_relative_score
#neg_mean_squared_error
def relative_score(y_true, y_pred):
		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -sum(res)/float(len(res))

def max_relative_score(y_true, y_pred):

		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -max(res)

#print solved percentage and avg solving only time
def printSvdPercAvgTime(p,runtime,maxtime,printresult=True):
	#success
	sucs=[]
	for i in runtime:
		if i<maxtime-1:
			sucs.append(i)
	if len(sucs)!=0:
		if printresult:
			print(p,float(len(sucs))/len(runtime),"/",float(sum(sucs))/len(sucs))
		return float(len(sucs))/len(runtime), float(sum(sucs))/len(sucs)
	else:
		if printresult:
			print(p,float(0),"/",float(0))
		return 0,0

#print solved percentage and real avg runtime
def printSvdPercAvgTime2(p,runtime,maxtime,printresult=True):
	#success
	sucs=[]
	time_real=[]
	for i in runtime:
		time_real.append(i)
		if i<maxtime-1:
			sucs.append(i)
	if len(sucs)!=0:
		if printresult:
			print(p,float(len(sucs))/len(runtime),"/",float(sum(time_real))/len(runtime))
		return float(len(sucs))/len(runtime), float(sum(time_real))/len(runtime)
	else:
		if printresult:
			print(p,float(0),"/",float(maxtime))
		return 0,0

#split 80% trainset into validSet, trainSet with specified binNum and which bin.
#bin=0, binNum=5.
#the last bin for validing, first 4bins for training.
def splitTrainValid(datasetX,bin,binNum):
	bin_size=int(math.ceil(len(datasetX)/binNum))
	if bin==0:
		return np.array(datasetX[bin_size:]),np.array(datasetX[:bin_size])
	elif bin==binNum-1:
		return np.array(datasetX[:(binNum-1)*bin_size]),np.array(datasetX[-bin_size:])
	else:
		return np.append(datasetX[:bin_size*(bin)],datasetX[bin_size*(bin+1):],axis=0),np.array(datasetX[bin_size*(bin):bin_size*(bin+1)])


def drawLine():
    print("------------------------------------------------")


def getfromindex(input_df,csvfile):
    df=pd.read_csv(csvfile)
    #print(input_df)
    #print(df)
    instance_value=df['Instance_index'].values
    #print(instance_value)
    return input_df.loc[instance_value]


#ml for each ml_group
#ml_last_group comes first, cause it has most encodings
def machine_learning(args,ml_group,ml_last_group):

    # get features and performances folders for each groups
    feature_folder=args.feature_folder[0]+'/'+ml_group
    performance_folder=args.performance_folder[0]+'/'+ml_group
    cutoff=args.cutoff[0]

    #output for all groups
    ml_outfolder=args.ml_models_folder[0]+'/'+ml_group
    ml_hyperfolder=args.ml_hyper_folder[0]+'/'+ml_group

    #output for last group
    #used for retrive train, test data, keeps consistency over groups
    ml_last_outfolder=args.ml_models_folder[0]+'/'+ml_last_group

    #make folders
    checkMakeFolder(ml_hyperfolder)
    checkMakeFolder(ml_outfolder)

    #set according to your cutoff time
    TIME_MAX=int(cutoff)
    #print('test time max',TIME_MAX)

    #use varing PENALTY policy PARX or fixed
    #False here, because it moved to candidate generation code
    PARX=True
    #set PENALTY_TIME
    PENALTY_TIME=int(cutoff)
    #seed for shuffle
    np.random.seed(3)
    random.seed(3 )

    score_functions=[make_scorer(relative_score),make_scorer(max_relative_score),"neg_mean_squared_error"]
    # here choose "neg_mean_squared_error"
    score_f=score_functions[2]

    ##data collection

    #combine features
    featureFile=feature_folder+'/'+os.listdir(feature_folder)[0]
    featureValue=pd.read_csv(featureFile)
    featureValue=featureValue.set_index(featureValue.columns[0])
    allCombine=featureValue.copy()

    #combine features and performance
    performanceFile=performance_folder+'/'+os.listdir(performance_folder)[0]
    performanceValue=pd.read_csv(performanceFile)
    performanceValue=performanceValue.set_index(performanceValue.columns[0])
    algorithmNames=performanceValue.columns.values
    performanceValue.columns=["runtime_"+algo for algo in algorithmNames]
    allCombine=allCombine.join(performanceValue)

    #remove duplicated
    allCombine = allCombine[~allCombine.index.duplicated(keep='first')]
    allCombine.sort_index()

    #print features
    featureList=allCombine.columns.values[:-len(algorithmNames)]
    print("[Feature used]:",featureList)

    #drop "na" rows
    allCombine=allCombine.dropna(axis=0, how='any')

    #drop "?" rows
    for feature in featureList[1:]:
        if allCombine[feature].dtypes=="object":
            # delete from the pd1 rows that contain "?"
            allCombine=allCombine[allCombine[feature].astype("str")!="?"]

    #oracle analysis
    algs=["runtime_"+algo for algo in algorithmNames]
    allRuntime=allCombine[algs]

    print(allRuntime.shape,allRuntime)
    oracle_value=np.amin(allRuntime.values, axis=1)
    oracle_index=np.argmin(allRuntime.values, axis=1)
    Oracle_name=[algorithmNames[oracle_index[i]] for i in range(len(oracle_index))]

    allCombine["Oracle_value"]=oracle_value
    allCombine["Oracle_name"]=Oracle_name
    allCombine["Instance_index"]=allCombine.index.values

    
    #varing penalty parx
    #moved to candidate selection code

    if PARX:      
        #Get how many enc timeouts each instance
        rt_tos=[]
        for idx in allRuntime.index:
            rts=allRuntime.loc[idx]
            #print('rts',rts)
            rt_tos.append(sum([int(ti)>TIME_MAX-1 for ti in rts]))
        #print('rt_tos',rt_tos)
        #Update runtime for enc timeouts for instances
        enc_values=algs
        for i_index,i in enumerate(allCombine.index):
            for j in enc_values:
                if int(allCombine.loc[i,j]) > TIME_MAX-1:
                    allCombine.loc[i,j]= rt_tos[i_index]*PENALTY_TIME
            
        #print(allCombine)


    #data split
    #last group split, other groups copy instance index
    if ml_group==ml_last_group:
        #shuffle
        allCombine=allCombine.iloc[np.random.permutation(len(allCombine))]

        # get leave out data 20% of the full data:
        leaveIndex=random.sample(range(allCombine.shape[0]), int(allCombine.shape[0]*0.2))
        mlIndex=list(range(allCombine.shape[0]))
        for i in leaveIndex:
            if i in mlIndex:
                mlIndex.remove(i)
        leaveSet=allCombine.iloc[leaveIndex]

        #remaining 75% as traing set
        trainSetAll=allCombine.iloc[mlIndex]

        print("ALL after preprocess:",len(trainSetAll)+len(leaveSet))
        print("Train set:",trainSetAll.shape)
        print("Test set:",leaveSet.shape)

        trainSetAll.to_csv(ml_outfolder+'/trainSetAll.csv')
        leaveSet.to_csv(ml_outfolder+"/leaveSet.csv")

    else:# keep data same as last group
        trainSetAll=getfromindex(allCombine,ml_last_outfolder+"/trainSetAll.csv")
        leaveSet=getfromindex(allCombine,ml_last_outfolder+"/leaveSet.csv")

        print("ALL after preprocess:",len(trainSetAll)+len(leaveSet))
        print("Train set:",trainSetAll.shape)
        print("Test set:",leaveSet.shape)       

        trainSetAll.to_csv(ml_outfolder+'/trainSetAll.csv')
        leaveSet.to_csv(ml_outfolder+"/leaveSet.csv")



    #train each model:
    #hyperparameters tuning
    #grid search
    bestDepth={}
    if os.path.isdir(ml_hyperfolder):
        pickleFiles=[pickFile for pickFile in os.listdir(ml_hyperfolder) if pickFile.endswith(".pickle")]
        if 'regression_bestDepth.pickle' in pickleFiles:
            with open(ml_hyperfolder+'/regression_bestDepth.pickle', 'rb') as handle:
                bestDepth = pickle.load(handle)



    for alg in algorithmNames:
        #hyperparameters tuning on whole dataset
        trainSetAll_X=trainSetAll.loc[:,featureList].values
        trainSetAll_y=trainSetAll["runtime_"+alg].values

        #load hyperparameters
        #if no hyperparameters tuned, hyperparameters tuning on whole dataset

        bestDepthDT=0
        bestDepthRF=0
        bestKNeib=0

        #load hyperparameters
        pickleFiles=[pickFile for pickFile in os.listdir(ml_hyperfolder) if pickFile.endswith(".pickle")]
        if 'regression_bestDepth.pickle' in pickleFiles:
            with open(ml_hyperfolder+'/regression_bestDepth.pickle', 'rb') as handle:
                bestDepth = pickle.load(handle)
                bestDepthDT,bestDepthRF,bestKNeib=bestDepth.get(alg,(0,0,0))

        #hyperparameters tuning if not exitst
        if bestKNeib==0 and bestDepthDT==0 and bestDepthRF==0:

            max_depth = range(2, 30, 1)
            dt_scores = []
            for k in max_depth:
                regr_k =tree.DecisionTreeRegressor(max_depth=k)
                loss = -cross_val_score(regr_k, trainSetAll_X, trainSetAll_y, cv=10, scoring=score_f)
                dt_scores.append(loss.mean())
            bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores,max_depth)))[0]


            max_depth = range(2, 30, 1)
            dt_scores = []
            for k in max_depth:
                regr_k = RandomForestRegressor(max_depth=k)
                loss = -cross_val_score(regr_k, trainSetAll_X, trainSetAll_y, cv=10, scoring=score_f)
                dt_scores.append(loss.mean())
            bestscoreRF,bestDepthRF=sorted(list(zip(dt_scores,max_depth)))[0]

            max_neigh = range(2, 30, 1)
            kNN_scores = []
            for k in max_neigh:
                kNeigh =KNeighborsRegressor(n_neighbors=k)
                loss = -cross_val_score(kNeigh,trainSetAll_X, trainSetAll_y, cv=10, scoring=score_f)
                kNN_scores.append(loss.mean())
            bestscoreKNN,bestKNeib=sorted(list(zip(kNN_scores,max_neigh)))[0]


            bestDepth[alg]=(bestDepthDT,bestDepthRF,bestKNeib)
            with open(ml_hyperfolder+'/regression_bestDepth.pickle', 'wb') as handle:
                pickle.dump(bestDepth, handle)

    #now we have three models which one to choose? use 5 folds cross validation
    # 5 folds cross validation to choose best out of 3 models
    crossvalidation_result={'DT':[],'RF':[],'kNN':[]}

    for valid_bin_split in range(5):
        # per folder
        print(valid_bin_split,'th fold validation:')
        #load trainSet, validSet
        trainSet,validSet=splitTrainValid(trainSetAll,valid_bin_split,5)
        trainSet=pd.DataFrame(trainSet,columns=trainSetAll.columns)
        validSet=pd.DataFrame(validSet,columns=trainSetAll.columns)

        trainResult=trainSet.copy()
        validResult=validSet.copy()

        #prediction results on trainSet and validSet
        for alg in algorithmNames:
            #get training result and validation result for each alg
            trainSet_X=trainSet.loc[:,featureList].values
            trainSet_y=trainSet["runtime_"+alg].values
            validSet_X=validSet.loc[:,featureList].values
            validSet_y=validSet["runtime_"+alg].values   

            #predict on all train and valid set and save result
            dtModel=tree.DecisionTreeRegressor(max_depth=bestDepthDT)
            dtModel= dtModel.fit(trainSet_X, trainSet_y)
            y_=dtModel.predict(trainSet_X)
            trainResult["DT_"+alg+"_pred"]=y_
            y_=dtModel.predict(validSet_X)
            validResult["DT_"+alg+"_pred"]=y_

            ##########
            rfModel=RandomForestRegressor(max_depth=bestDepthRF)
            rfModel= rfModel.fit(trainSet_X, trainSet_y)
            y_=rfModel.predict(trainSet_X)
            trainResult["RF_"+alg+"_pred"]=y_
            y_=rfModel.predict(validSet_X)
            validResult["RF_"+alg+"_pred"]=y_

            #########
            kNeigh =KNeighborsRegressor(n_neighbors=bestKNeib)
            kNeigh= kNeigh.fit(trainSet_X, trainSet_y)
            y_=kNeigh.predict(trainSet_X)
            trainResult["kNN_"+alg+"_pred"]=y_
            y_=kNeigh.predict(validSet_X)
            validResult["kNN_"+alg+"_pred"]=y_

        #analysis
        ##solved percent and runtime of each validation 
        ##per algorithm, oracle and ES

        ##training  result
        runtimeIndex=[i for i in trainResult.columns if "runtime" in i]
        drawLine()
        print("trainSet")
        print("Indivadual encoding and Oracle performance: ")
        #print per algorithm
        for alg in runtimeIndex:
            printSvdPercAvgTime(alg.split("_")[1],trainResult[alg],TIME_MAX)
        #print oracle
        printSvdPercAvgTime("oracle_portfolio",trainResult.Oracle_value.values,TIME_MAX)
        print("\nEncoding selection performance:")
        for mName in "DT,RF,kNN".split(","):

            print(mName)
            encRuntime=[i for i in trainResult.columns if "runtime" in i]
            modelRuntime=[i for i in trainResult.columns if mName in i]
            modelResults=trainResult[encRuntime+modelRuntime].copy()

            #save each instance's predicted runtime of six encoding and the corresponding predicted encoding name
            #(runtime, name)
            #for each instance, sort by runtime, so that we know which is the first predicted one
            modelResultsCopy=modelResults[modelRuntime].copy()
            for i in modelResultsCopy.columns.values:
                modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
            predictedList=modelResultsCopy.values
            predictedList.sort()

            #the best predicted is the i[0]:(min_runtime, its_name)
            bestpredname=[i[0][1] for i in predictedList]
            bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
            bestruntime=[modelResults[bestname[i]].values[i]  for i in range(len(modelResults))]
            modelResults["1st_ham"]=bestname
            modelResults["1st_time"]=bestruntime
            printSvdPercAvgTime("1st",bestruntime,TIME_MAX)
            

        #validation results
        
        print("\n")
        print("Validation Set")  
        drawLine()
        print("Indivadual encoding and Oracle performance: ")
        for alg in runtimeIndex:
            printSvdPercAvgTime(alg+"",validResult[alg],TIME_MAX)
        printSvdPercAvgTime("oracle_portfolio",validResult.Oracle_value.values,TIME_MAX)
        print("\nEncoding selection performance: ")
        for mName in "DT,RF,kNN".split(","):
            print(mName)
            encRuntime=[i for i in validResult.columns if "runtime" in i]
            modelRuntime=[i for i in validResult.columns if mName in i]
            modelResults=validResult[encRuntime+modelRuntime].copy()

            modelResultsCopy=modelResults[modelRuntime].copy()
            for i in modelResultsCopy.columns.values:
                modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
            predictedList=modelResultsCopy.values
            predictedList.sort()

            bestpredname=[i[0][1] for i in predictedList]
            bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
            bestruntime=[modelResults[bestname[i]].values[i]  for i in range(len(modelResults))]
            modelResults["1st_ham"]=bestname
            modelResults["1st_time"]=bestruntime
            sv_percent,sv_time=printSvdPercAvgTime("1st",bestruntime,TIME_MAX)

            #record each k folder crossvalidation results
            crossvalidation_result[mName].append((sv_percent,sv_time))
                
    print(crossvalidation_result)
    #get best models out of three modles with  5 folder crossvalidation result
    #get avg of each model of 5 folder
    validResultSaving=[]
    for model_name in crossvalidation_result:
        solving_per_all_k=0
        solving_time_all_k=0
        for k_fold_result in crossvalidation_result[model_name]:
            solving_per_all_k+=k_fold_result[0]
            solving_time_all_k+=k_fold_result[1]
        validResultSaving.append((solving_per_all_k/5,solving_time_all_k/5,model_name))
    #sort by avg solving per
    #get best out of 3 models
    validResultSaving=sorted(validResultSaving)[-1]
    print(validResultSaving)
    method=str(validResultSaving[2])
    result_sol=str(validResultSaving[0])
    result_tm=str(validResultSaving[1])

    #write to evaluation result of each group
    with open('evaluation/result.csv','a') as f:
        f.write(method+'_'+ml_group+','+result_sol+','+result_tm+'\n')

    print('\n')

    #now we have best models for each group 
    #we test all models on leave out set
    #leave out set; test set
    #load best model on whole trainSet and test on leave out set;
    #other models also print, work as comparison

    trainSet=trainSetAll.copy()
    trainResult=trainSet.copy()

    validSet=leaveSet.copy()
    validResult=validSet.copy()

    #prediction results on trainSet and validSet
    for alg in algorithmNames:
        #get training result and validation result for each alg
        trainSet_X=trainSet.loc[:,featureList].values
        trainSet_y=trainSet["runtime_"+alg].values
        validSet_X=validSet.loc[:,featureList].values
        validSet_y=validSet["runtime_"+alg].values   

        #predict on all train and valid set and save result
        dtModel=tree.DecisionTreeRegressor(max_depth=bestDepthDT)
        dtModel= dtModel.fit(trainSet_X, trainSet_y)
        y_=dtModel.predict(trainSet_X)
        trainResult["DT_"+alg+"_pred"]=y_
        y_=dtModel.predict(validSet_X)
        validResult["DT_"+alg+"_pred"]=y_

        ##########
        rfModel=RandomForestRegressor(max_depth=bestDepthRF)
        rfModel= rfModel.fit(trainSet_X, trainSet_y)
        y_=rfModel.predict(trainSet_X)
        trainResult["RF_"+alg+"_pred"]=y_
        y_=rfModel.predict(validSet_X)
        validResult["RF_"+alg+"_pred"]=y_

        #########
        kNeigh =KNeighborsRegressor(n_neighbors=bestKNeib)
        kNeigh= kNeigh.fit(trainSet_X, trainSet_y)
        y_=kNeigh.predict(trainSet_X)
        trainResult["kNN_"+alg+"_pred"]=y_
        y_=kNeigh.predict(validSet_X)
        validResult["kNN_"+alg+"_pred"]=y_

    #analysis
    ##solved percent and runtime of each validation 
    ##per algorithm, oracle and ES

    ##training  result
    runtimeIndex=[i for i in trainResult.columns if "runtime" in i]
    drawLine()
    print("trainSetAll")
    print("Indivadual encoding and Oracle performance: ")
    #print per algorithm
    for alg in runtimeIndex:
        printSvdPercAvgTime(alg.split("_")[1],trainResult[alg],TIME_MAX)
    #print oracle
    printSvdPercAvgTime("oracle_portfolio",trainResult.Oracle_value.values,TIME_MAX)
    print("\nEncoding selection performance:")
    for mName in "DT,RF,kNN".split(","):

        print(mName)
        encRuntime=[i for i in trainResult.columns if "runtime" in i]
        modelRuntime=[i for i in trainResult.columns if mName in i]
        modelResults=trainResult[encRuntime+modelRuntime].copy()

        #save each instance's predicted runtime of six encoding and the corresponding predicted encoding name
        #(runtime, name)
        #for each instance, sort by runtime, so that we know which is the first predicted one
        modelResultsCopy=modelResults[modelRuntime].copy()
        for i in modelResultsCopy.columns.values:
            modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
        predictedList=modelResultsCopy.values
        predictedList.sort()

        #the best predicted is the i[0]:(min_runtime, its_name)
        bestpredname=[i[0][1] for i in predictedList]
        bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
        bestruntime=[modelResults[bestname[i]].values[i]  for i in range(len(modelResults))]
        modelResults["1st_ham"]=bestname
        modelResults["1st_time"]=bestruntime
        printSvdPercAvgTime("1st",bestruntime,TIME_MAX)
        

    #validation results
    
    print("\n")
    #print("Test Set")  
    drawLine()
    print("Indivadual encoding and Oracle performance: ")
    for alg in runtimeIndex:
        sv_percent,sv_time=printSvdPercAvgTime(alg+"",validResult[alg],TIME_MAX,False)
        write2eva2(alg+ml_group,sv_percent,sv_time)
    sv_percent,sv_time=printSvdPercAvgTime("oracle_portfolio",validResult.Oracle_value.values,TIME_MAX,False)
    write2eva2("oracle_portfolio"+ml_group,sv_percent,sv_time)
    #print("\nEncoding selection performance: ")
    for mName in "DT,RF,kNN".split(","):
        #print(mName)
        encRuntime=[i for i in validResult.columns if "runtime" in i]
        modelRuntime=[i for i in validResult.columns if mName in i]
        modelResults=validResult[encRuntime+modelRuntime].copy()

        modelResultsCopy=modelResults[modelRuntime].copy()
        for i in modelResultsCopy.columns.values:
            modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
        predictedList=modelResultsCopy.values
        predictedList.sort()

        bestpredname=[i[0][1] for i in predictedList]
        bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
        bestruntime=[modelResults[bestname[i]].values[i]  for i in range(len(modelResults))]
        modelResults["1st_ham"]=bestname
        modelResults["1st_time"]=bestruntime
        sv_percent,sv_time=printSvdPercAvgTime("1st",bestruntime,TIME_MAX,False)
        write2eva2(mName+ml_group,sv_percent,sv_time)


if __name__ == "__main__":
    print('\nMachine learning model building...')
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()


    ml_outfolder=args.ml_models_folder[0]
    ml_hyperfolder=args.ml_hyper_folder[0]

    checkMakeFolder(ml_hyperfolder)
    checkMakeFolder(ml_outfolder)

    if check_content(ml_hyperfolder) or check_content(ml_hyperfolder):
        cleanFolder([ml_hyperfolder,ml_outfolder])

    #evaluating
    if not os.path.exists('evaluation'):
        os.system('mkdir evaluation')
    os.system('rm evaluation/*')
    with open('evaluation/result.csv','w') as f:
        f.write('method,solving,time\n')    

    with open('evaluation/result2.csv','w') as f:
        f.write('test\n')

    feature_folder=args.feature_folder[0]
    feature_groups=os.listdir(feature_folder)
    for ml_group in sorted(feature_groups)[::-1]:
        machine_learning(args,ml_group,sorted(feature_groups)[-1])
    
