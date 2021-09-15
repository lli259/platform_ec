import argparse,os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def define_args(arg_parser):

    arg_parser.add_argument('--feature_folder', nargs='*', default=['features'], help='Gringo input files')
    arg_parser.add_argument('--feature_outfolder', nargs='*', default=['features_selected'], help='Gringo input files')
    arg_parser.add_argument('--feature_folder_extra', nargs='*', default=['features_domain'], help='Gringo input files')
    arg_parser.add_argument('--performance_folder', nargs='*', default=['performance_selected'], help='Gringo input files')
    


def get_most_meaningful(feature_data,performance_data):
    alldata=feature_data.join(performance_data)

    cols=alldata.columns.values
    #print(alldata.shape)
    alldata=alldata.dropna()
    #print(alldata.shape)
    X_Train=alldata.loc[:,cols[:-1]]
    Y_Train=alldata.loc[:,cols[-1:]]
    #exit()
    number_features=min(int(X_Train.shape[1]/3),int(X_Train.shape[0]/10))
    if number_features <=10:
        number_features=min(10,X_Train.shape[1])
    #print(X_Train.shape,Y_Train.shape)
    #X_Train = StandardScaler().fit_transform(X_Train)
    #print(X_Train,Y_Train)
    #print(X_Train.shape,Y_Train.shape)
    Y_Train=Y_Train.values.reshape(X_Train.shape[0],)
    
    trainedforest = RandomForestRegressor(n_estimators=200,max_depth=20).fit(X_Train,Y_Train)

    feat_importances = pd.Series(trainedforest.feature_importances_, index= X_Train.columns)
    select_f=feat_importances.nlargest(number_features).index.values
    return select_f

def get_accuracy(most_meaning_f,feature_data,performance_data):
    alldata=feature_data.join(performance_data)
    alldata=alldata.dropna()
    cols=alldata.columns.values
    X_Train=alldata.loc[:,most_meaning_f]
    Y_Train=alldata.loc[:,cols[-1:]]

    #X_Train = StandardScaler().fit_transform(X_Train)
    Y_Train=Y_Train.values.reshape(X_Train.shape[0],)
    trainedforest = RandomForestRegressor(n_estimators=200,max_depth=20).fit(X_Train,Y_Train)
    predictionforest = trainedforest.predict(X_Train)
    
    return mean_squared_error(Y_Train,predictionforest)

def save_to_folder(args,selected_features,selected_file):
    feature_outfolder=args.feature_outfolder[0]
    if not os.path.exists(feature_outfolder):
        os.system('mkdir '+feature_outfolder)


    feature_folder=args.feature_folder[0]

    feature_data=pd.read_csv(feature_folder+'/'+selected_file)
    feature_data=feature_data.set_index(feature_data.columns[0])    

    feature_data_selected=feature_data[selected_features]
    feature_data_selected.to_csv(feature_outfolder+'/'+'features_select.csv')


def save_to_folder_with_domain(args,selected_features,selected_file,most_meaning_f_dm):
    feature_outfolder=args.feature_outfolder[0]
    if not os.path.exists(feature_outfolder):
        os.system('mkdir '+feature_outfolder)


    feature_folder=args.feature_folder[0]
    feature_data=pd.read_csv(feature_folder+'/'+selected_file)
    feature_data=feature_data.set_index(feature_data.columns[0])    
    feature_data_selected=feature_data[selected_features]


    feature_domain_folder=args.feature_folder_extra[0]
    feature_domain_file=os.listdir(feature_domain_folder)[0]
    feature_domain=pd.read_csv(feature_domain_folder+'/'+feature_domain_file)
    feature_domain=feature_domain.set_index(feature_domain.columns[0])
    feature_domain_selected=feature_domain[most_meaning_f_dm]

    feature_data_selected=feature_data_selected.join(feature_domain_selected)
    feature_data_selected=feature_data_selected.dropna()
    feature_data_selected.to_csv(feature_outfolder+'/'+'features_select.csv')

def select_f(args):
    
    #clasp features
    feature_folder=args.feature_folder[0]
    performance_folder=args.performance_folder[0]

    feature_all_enc=os.listdir(feature_folder)
    performance_file=performance_folder+'/'+os.listdir(performance_folder)[0]

    performance_data=pd.read_csv(performance_file)
    performance_data=performance_data.set_index(performance_data.columns[0])
    performance_data=performance_data[performance_data.columns[0]]

    allscore=[]
    for f_each_enc in feature_all_enc:
        feature_data=pd.read_csv(feature_folder+'/'+f_each_enc)
        feature_data=feature_data.set_index(feature_data.columns[0])
        most_meaning_f=get_most_meaningful(feature_data,performance_data)
        score=get_accuracy(most_meaning_f,feature_data,performance_data)
        allscore.append((score,most_meaning_f,f_each_enc))
    allscore=sorted(allscore)

    selected_features=allscore[-1][1]
    selected_file=allscore[-1][2]

    #domain features
    feature_domain_folder=args.feature_folder_extra[0]
    feature_domain_file=os.listdir(feature_domain_folder)[0]
    if feature_domain_file ==None:
        save_to_folder(args,selected_features,selected_file)
    else:
        feature_domain=pd.read_csv(feature_domain_folder+'/'+feature_domain_file)
        feature_domain=feature_domain.set_index(feature_domain.columns[0])
        feature_domain=feature_domain.dropna()
        #print('domain feature selection...',feature_domain.shape)
        most_meaning_f_dm=get_most_meaningful(feature_domain,performance_data)
        save_to_folder_with_domain(args,selected_features,selected_file,most_meaning_f_dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()

    select_f(args)

    
    






