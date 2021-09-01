import argparse,os


'''
Platform Learning

'''
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




def define_args(arg_parser):
    arg_parser.description = 'ASP Platform'

    arg_parser.add_argument('-p', nargs='*', default=[], help='Gringo input files')
    arg_parser.add_argument('--encodings', nargs='*', default=['encodings'], help='Gringo input files')
    arg_parser.add_argument('--selected_encodings', nargs='*', default=['encodings_selected'], help='Gringo input files')
    arg_parser.add_argument('--instances', nargs='*', default=['instances'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')
    arg_parser.add_argument('--performance_data', nargs='*', default=['performance'], help='Gringo input files')
    arg_parser.add_argument('--performance_select', nargs='*', default=['performance_selected'], help='Gringo input files')   
    arg_parser.add_argument('--num_candidate', nargs='*', default=['4'], help='Gringo input files')
    arg_parser.add_argument('--feature_data', nargs='*', default=['features'], help='Gringo input files')
    arg_parser.add_argument('--feature_selected', nargs='*', default=['features_selected'], help='Gringo input files')
    arg_parser.add_argument('--ml_models_folder', nargs='*', default=['ml_models'], help='Gringo input files')    
    arg_parser.add_argument('--interleave_folder', nargs='*', default=['interleave'], help='Gringo input files') 
    arg_parser.add_argument('--schedule_folder', nargs='*', default=['schedule'], help='Gringo input files') 

parser = argparse.ArgumentParser()
define_args(parser)
args = parser.parse_args()

#rewrite

if args.p== [] or Encoding_rewrite in args.p:
    for enc_file in os.listdir(args.encodings[0]):
        if (not enc_file ==  None) and (not '_rewritten.lp' in enc_file):
            os.system('python aaggrewrite.py '+args.encodings[0]+'/'+enc_file)


if args.p== [] or Performance_gen in args.p:
    os.system('python performance_gen.py '
    +' --encodings ' +args.encodings[0]
    +' --instances ' +args.instances[0]
    +' --cutoff ' + args.cutoff[0]
    +' --performance_data ' + args.performance_data[0])


if args.p== [] or Encoding_candidate_gen in args.p:

    allcandidate=len(os.listdir(args.encodings[0]))
    #print('selected_candidate_number',args.num_candidate[0])
    selected_candidate_number=min(allcandidate,int(args.num_candidate[0]))
    print('selected_candidate_number',selected_candidate_number)
    os.system('python selected_candidate.py --num_candidate '+ str(selected_candidate_number) 
    +' --cutoff ' + args.cutoff[0]
    +' --performance_data ' + args.performance_data[0])


if args.p== [] or Feature_extraction in args.p:
    instances_folder=args.selected_encodings[0]
    encodings_folder=args.encodings[0]
    os.system('python2 feature_extract.py --instances_folder '+ instances_folder
    +' --encodings_folder ' + encodings_folder
    )

if args.p== [] or Feature_selection in args.p:
    feature_folder=args.feature_data[0]
    performance_folder=args.performance_select[0]

    os.system('python feature_selection.py --feature_folder '+ feature_folder
    +' --performance_folder ' + performance_folder
    )


if args.p== [] or Model_building in args.p:
    feature_folder=args.feature_selected[0]
    performance_folder=args.performance_select[0]
    cutoff=args.cutoff[0]


    os.system('python model_building.py --feature_folder '+ feature_folder 
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )


if args.p== [] or Schedule_building in args.p:

    performance_folder=args.performance_select[0]
    cutoff=args.cutoff[0]


    os.system('python schedule_build.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )

if args.p== [] or Interleaving_building in args.p:

    performance_folder=args.performance_select[0]
    cutoff=args.cutoff[0]


    os.system('python interleave_build.py '
    +' --performance_folder ' + performance_folder
    +' --cutoff ' + cutoff
    )
