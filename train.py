import argparse,os
import pandas as pd

'''
Platform Learning

'''
ALLRUN=['0']
Encoding_rewrite='1'
Performance_gen='2'
Encoding_candidate_gen='3'
Feature_extraction='4'
Feature_selection='5'
Model_building='6'
Schedule_building='7'
Interleaving_building='8'
Evaluation='9'

'''
Platform Prediction
use solve.py
'''

class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def define_args(arg_parser):
    arg_parser.description = 'ASP Platform'

    arg_parser.add_argument('-p', nargs='*', default=[], help='R|Platform process number\n'+
    '-p 0 :ALLRUN\n'+
    '-p 1 :Encoding_rewrite\n'+
    '-p 2 :Performance_gen\n'+
    '-p 3 :Encoding_candidate_gen\n'+
    '-p 4 :Feature_extraction\n'+
    '-p 5 :Feature_selection\n'+
    '-p 6 :Model_building\n'+
    '-p 7 :Schedule_building\n'+       
    '-p 8 :Interleaving_building\n'+
    '-p 9 :Evaluation\n')
    
    arg_parser.add_argument('--encodings', nargs='*', default=['encodings'], help='Platform input encodings folder')
    arg_parser.add_argument('--instances', nargs='*', default=['instances'], help='Gringo input instances folder')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Solving cutoff time')
    arg_parser.add_argument('--selected_encodings', nargs='*', default=['encodings_selected'], help='Platform selected encodings')
    arg_parser.add_argument('--rewrite_form', nargs='*', default=['1'], help='Rewrite form 1..4')
    arg_parser.add_argument('--performance_data', nargs='*', default=['performance'], help='Performance data folder')
    arg_parser.add_argument('--performance_select', nargs='*', default=['performance_selected'], help='Performance selected folder')   
    arg_parser.add_argument('--feature_data', nargs='*', default=['features'], help='Platform claspre feature folder')
    arg_parser.add_argument('--feature_domain', nargs='*', default=['features_domain'], help='Platform domain feature folder')
    arg_parser.add_argument('--feature_selected', nargs='*', default=['features_selected'], help='Feature selected folder')
    arg_parser.add_argument('--ml_models_folder', nargs='*', default=['ml_models'], help='ML models folder')    
    arg_parser.add_argument('--interleave_folder', nargs='*', default=['interleave'], help='Interleave schedule folder') 
    arg_parser.add_argument('--schedule_folder', nargs='*', default=['schedule'], help='Schedule folder')   
    arg_parser.add_argument('--performance_provided',action='store_true', help='Run all excluding performance collection') 
    arg_parser.add_argument('--perform_feat_provided',action='store_true', help='Run all excluding performance and feature collection') 

parser = argparse.ArgumentParser(description='esp_helper',formatter_class=SmartFormatter)
define_args(parser)
args = parser.parse_args()

#Encoding rewrite

if args.p== ALLRUN or Encoding_rewrite in args.p :
    for enc_file in os.listdir(args.encodings[0]):
        if (not enc_file ==  None) and (not 'aagg.lp'  in enc_file):
            os.system('python aaggrewrite.py '+args.encodings[0]+'/'+enc_file
            +' --aggregate_form ' + args.rewrite_form[0]
            )


#performance data generation
if args.p== ALLRUN or Performance_gen in args.p:
    os.system('python performance_gen.py '
    +' --encodings ' +args.encodings[0]
    +' --instances ' +args.instances[0]
    +' --cutoff ' + args.cutoff[0]
    +' --performance_data ' + args.performance_data[0])

    #Too hard or too easy instancss.
    if not os.path.exists('cutoff/cutoff.txt'):
        print('Data collection failed!')
        exit()

    cutoff_set=0
    with open('cutoff/cutoff.txt','r') as f:
        line=f.readline()
        cutoff_set=line
    #if passed, check if enough hard instance >500 for training
    allCombine_test=pd.read_csv(args.performance_data[0]+'/performance.csv')
    allCombine_testhard=allCombine_test.set_index('inst')
    #allCombine_testhard=allCombine_testhard.iloc[100:105,:3]
    #print(len(allCombine_testhard))

    all_hard=set()
    all_easy=set(allCombine_testhard.index.values)
    all_to=set(allCombine_testhard.index.values)
    for col in allCombine_testhard.columns.values:
        #union all hard: one hard is ok.
        no_easy_df=allCombine_testhard[allCombine_testhard[col]>float(cutoff_set)/7]
        hard_df=no_easy_df[no_easy_df[col]<float(cutoff_set)-1]
        all_hard.update(set(hard_df.index.values))

        #intersection on easy and to: all to or all easy
        easy_df=allCombine_testhard[allCombine_testhard[col]<float(cutoff_set)/7] 
        all_easy=all_easy.intersection(set(easy_df.index.values))  

        to_df=allCombine_testhard[allCombine_testhard[col]>float(cutoff_set)-1] 
        all_to=all_to.intersection(set(to_df.index.values))

    #print(all_hard)
    #print(len(all_hard))
    if len(all_hard)<500:
        print('Less than 500 hard instances:',len(all_hard),'! Add more instances!')   
        print('Found easy instances:',len(all_easy))
        print('Found Timeout instances:',len(all_to))
        with open ('cutoff/cutoff.txt','r',) as f:
            cutset=f.readline()
            print('Cutoff set as',cutset)
        exit()


#Encoding_candidate generation
if args.p== ALLRUN or Encoding_candidate_gen in args.p or args.performance_provided or args.perform_feat_provided:

    cutoff=args.cutoff[0]

    if Performance_gen in args.p and os.path.exists('cutoff/cutoff.txt'):
        with open('cutoff/cutoff.txt','r') as f:
            cutoff=f.readline()
    #encoding candidates are not selected if only 1, or 0 encoding
    encodings_all=os.listdir(args.encodings[0])
    if len(encodings_all) < 2:
        print('Less than two encodings! Provide more encodings!')
        exit()
     
    os.system('python selected_candidate.py '
    +' --encodings ' +args.encodings[0]
    +' --selected_encodings ' +args.selected_encodings[0]
    +' --cutoff ' + cutoff
    +' --performance_data ' + args.performance_data[0])

    df=pd.read_csv(args.performance_data[0]+"_output/allwins.csv")
    df_win_name=df['win_name'].values
    if len(set(df_win_name)) ==1:
        print('All the winners are the same encoding! Provide other encodings or more instances!')
        exit()

#Feature extraction
if args.p== ALLRUN or Feature_extraction in args.p or args.performance_provided:
    instances_folder=args.instances[0]
    encodings_folder=args.encodings[0]
    os.system('python feature_extract.py --instances_folder '+ instances_folder
    +' --encodings_folder ' + encodings_folder
    )

#Feature selection
if args.p== ALLRUN or Feature_selection in args.p or args.performance_provided or args.perform_feat_provided:
    feature_folder=args.feature_data[0]
    performance_folder=args.performance_select[0]
    feature_folder_extra = args.feature_domain[0]
    os.system('python feature_selection.py --feature_folder '+ feature_folder
    +' --feature_folder_extra ' + feature_folder_extra
    +' --performance_folder ' + performance_folder
    )

#Machine Learning Model building
if args.p== ALLRUN or Model_building in args.p or args.performance_provided or args.perform_feat_provided:
    feature_folder=args.feature_selected[0]
    performance_folder=args.performance_select[0]
    #cutoff=args.cutoff[0]
    #cutoff ='200'

    cutoff=args.cutoff[0]

    if Performance_gen in args.p and os.path.exists('cutoff/cutoff.txt'):
        with open('cutoff/cutoff.txt','r') as f:
            cutoff=f.readline()

    os.system('python model_building.py --feature_folder '+ feature_folder 
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )

#Schedule building
if args.p== ALLRUN or Schedule_building in args.p or args.performance_provided or args.perform_feat_provided:

    performance_folder=args.performance_select[0]
    
    #cutoff ='200'

    cutoff=args.cutoff[0]

    if Performance_gen in args.p and os.path.exists('cutoff/cutoff.txt'):
        with open('cutoff/cutoff.txt','r') as f:
            cutoff=f.readline()

    os.system('python schedule_build.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )

#Interleaving Schedule building
if args.p== ALLRUN or Interleaving_building in args.p or args.performance_provided or args.perform_feat_provided:

    performance_folder=args.performance_select[0]

    #cutoff ='200'
    cutoff=args.cutoff[0]

    if Performance_gen in args.p and os.path.exists('cutoff/cutoff.txt'):
        with open('cutoff/cutoff.txt','r') as f:
            cutoff=f.readline()

    os.system('python interleave_build.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )


#Interleaving Schedule building
if args.p== ALLRUN or Evaluation in args.p or args.performance_provided or args.perform_feat_provided:

    performance_folder=args.performance_select[0]

    cutoff=args.cutoff[0]

    if Performance_gen in args.p and os.path.exists('cutoff/cutoff.txt'):
        with open('cutoff/cutoff.txt','r') as f:
            cutoff=f.readline()

    os.system('python evaluation.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )
