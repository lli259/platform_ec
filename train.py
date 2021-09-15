import argparse,os


'''
Platform Learning

'''
Encoding_rewrite='1'
#Performance_gen='2'
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




def define_args(arg_parser):
    arg_parser.description = 'ASP Platform'

    arg_parser.add_argument('-p', nargs='*', default=[], help='Platform process number')
    arg_parser.add_argument('--encodings', nargs='*', default=['encodings'], help='Platform input encodings')
    arg_parser.add_argument('--selected_encodings', nargs='*', default=['encodings_selected'], help='Platform selected encodings')
    arg_parser.add_argument('--instances', nargs='*', default=['instances'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')
    arg_parser.add_argument('--rewrite_form', nargs='*', default=['0'], help='Gringo input files')
    arg_parser.add_argument('--performance_data', nargs='*', default=['performance'], help='Gringo input files')
    arg_parser.add_argument('--performance_select', nargs='*', default=['performance_selected'], help='Gringo input files')   
    arg_parser.add_argument('--num_candidate', nargs='*', default=['4'], help='Gringo input files')
    arg_parser.add_argument('--feature_data', nargs='*', default=['features'], help='Gringo input files')
    arg_parser.add_argument('--feature_domain', nargs='*', default=['features_domain'], help='Gringo input files')
    arg_parser.add_argument('--feature_selected', nargs='*', default=['features_selected'], help='Gringo input files')
    arg_parser.add_argument('--ml_models_folder', nargs='*', default=['ml_models'], help='Gringo input files')    
    arg_parser.add_argument('--interleave_folder', nargs='*', default=['interleave'], help='Gringo input files') 
    arg_parser.add_argument('--schedule_folder', nargs='*', default=['schedule'], help='Gringo input files') 
    arg_parser.add_argument('--preprocessed', nargs='*', default=['0'], help='Gringo input files') 

parser = argparse.ArgumentParser()
define_args(parser)
args = parser.parse_args()

#Encoding rewrite

if args.p== [0] or Encoding_rewrite in args.p :
    for enc_file in os.listdir(args.encodings[0]):
        if (not enc_file ==  None) and (not 'aagg.lp' in enc_file):
            os.system('python aaggrewrite.py '+args.encodings[0]+'/'+enc_file)

'''
#performance data generation
if args.p== [0] or Performance_gen in args.p:
    os.system('python performance_gen.py '
    +' --encodings ' +args.encodings[0]
    +' --instances ' +args.instances[0]
    +' --cutoff ' + args.cutoff[0]
    +' --performance_data ' + args.performance_data[0])
'''


#Encoding_candidate generation
if args.p== [0] or Encoding_candidate_gen in args.p:

    allcandidate=len(os.listdir(args.encodings[0]))
    #print('selected_candidate_number',args.num_candidate[0])
    selected_candidate_number=min(allcandidate,int(args.num_candidate[0]))
    os.system('python selected_candidate.py --num_candidate '+ str(selected_candidate_number) 
    +' --encodings ' +args.encodings[0]
    +' --selected_encodings ' +args.selected_encodings[0]
    +' --cutoff ' + args.cutoff[0]
    +' --performance_data ' + args.performance_data[0])
    print('selected_candidate_number:',selected_candidate_number)

#Feature extraction
if args.p== [0] or Feature_extraction in args.p:
    instances_folder=args.instances[0]
    encodings_folder=args.selected_encodings[0]
    os.system('python2 feature_extract.py --instances_folder '+ instances_folder
    +' --encodings_folder ' + encodings_folder
    )

#Feature selection
if args.p== [0] or Feature_selection in args.p:
    feature_folder=args.feature_data[0]
    performance_folder=args.performance_select[0]
    feature_folder_extra = args.feature_domain[0]
    os.system('python feature_selection.py --feature_folder '+ feature_folder
    +' --feature_folder_extra ' + feature_folder_extra
    +' --performance_folder ' + performance_folder
    )

#Machine Learning Model building
if args.p== [0] or Model_building in args.p or '1' in args.preprocessed:
    feature_folder=args.feature_selected[0]
    performance_folder=args.performance_select[0]
    #cutoff=args.cutoff[0]
    cutoff ='200'

    os.system('python model_building.py --feature_folder '+ feature_folder 
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )

#Schedule building
if args.p== [0] or Schedule_building in args.p or '1' in args.preprocessed:

    performance_folder=args.performance_select[0]
    #cutoff=args.cutoff[0]
    cutoff='200'

    os.system('python schedule_build.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )

#Interleaving Schedule building
if args.p== [0] or Interleaving_building in args.p or '1' in args.preprocessed:

    performance_folder=args.performance_select[0]
    #cutoff=args.cutoff[0]
    cutoff='200'

    os.system('python interleave_build.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )


#Interleaving Schedule building
if args.p== [0] or Evaluation in args.p or '1' in args.preprocessed:

    performance_folder=args.performance_select[0]
    #cutoff=args.cutoff[0]
    cutoff='200'

    os.system('python evaluation.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )
