import argparse,os
import numpy as np
from collections import Counter
import pandas as pd
import subprocess

def define_args(arg_parser):

    arg_parser.add_argument('--encodings', nargs='*', default=['encodings'], help='Gringo input files')
    arg_parser.add_argument('--instances', nargs='*', default=['instances'], help='Gringo input files')    
    arg_parser.add_argument('--performance_data', nargs='*', default=['performance'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')


def encoding_name_parser(enc_name):
    return enc_name.split('.')[0].split('_')[0]

def clasp_result_parser(outputs):

    outputs=outputs.split('\n')
    re_time=0
    re_model=0
    for lineout in outputs:
        if 'Time' in lineout[:4]:
            re_time=lineout.split(':')[1].split('s')[0][1:]
        if 'Models' in lineout[:6]:
            re_model=lineout.split(':')[1][1:]
    return re_time,re_model

def getins(infile):

    if not os.path.isfile(infile):
        return []
    ret=[]
    with open(infile,'r') as f:
        lines=f.readlines()
        if len(lines)<2:
            return []
        for l in lines[1:]:
            ret.append(l.split(",")[0])
    return ret

def get_solved_instance(outfile):

    run_inst=getins(outfile)

    if len(run_inst)==0:
        with open(outfile,'w') as f:
            f.write('inst,time,model\n')
    return run_inst

def run_instances_for_enc(enc_name,encodings_folder,instances_names,instances_folder,out_folder,cutoff_t):

    #check if enc_result.csv exist, 
    #if exists, get instances
    #if not, create
    outfile=out_folder+'/'+encoding_name_parser(enc_name)+'_result.csv'
    solved_instances=get_solved_instance(outfile)

    print('solving instances using '+enc_name)


    for instance in instances_names:  
        if not instance.split(".")[0] in solved_instances:
            cmdline='tools/gringo '+encodings_folder+'/'+enc_name +' '+instances_folder+'/'+instance +' | tools/clasp --time-limit=' + str(cutoff_t)
            print(cmdline)
            #print('Solving ',instances_folder+'/'+instance)
            process = subprocess.getoutput(cmdline)
            #getoutput
            tm,md=clasp_result_parser(process)
            print(tm,md)
            with open (outfile,'a') as f:
                f.write(str(instance).split('.')[0]+','+str(tm)+','+str(md)+'\n')            



def combine_result(data_folder1,data_folder2):

    output_file='performance.csv'
    allcsv=os.listdir(data_folder1)

    pd1=pd.read_csv(data_folder1+'/'+allcsv[0])
    cols=pd1.columns.values
    pd1=pd1.set_index(cols[0])
    timecol='time'
    pd1=pd1[[timecol]]
    cols=[ timecol+'_'+allcsv[0].split('_')[0] ]
    pd1.columns=cols

    for pointer in range(1,len(allcsv)):
        print(pointer,len(allcsv))
        pd11=pd.read_csv(data_folder1+'/'+allcsv[pointer])
        cols=pd11.columns.values
        pd11=pd11.set_index(cols[0])
        timecol='time'
        pd11=pd11[[timecol]]
        cols=[ timecol+'_'+allcsv[pointer].split('_')[0] ]
        pd11.columns=cols
        
        pd1=pd1.join(pd11)
        pd1=pd1.dropna()
    pd1.to_csv(data_folder2+'/'+output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()


    encodings_folder=args.encodings[0]
    instances_folder=args.instances[0]
    t_cutoff=int(args.cutoff[0])
    data_final=args.performance_data[0]
    output_result_folder=data_final+'_each_enc'

    if not os.path.exists(data_final):
        os.mkdir(data_final)    


    if not os.path.exists(output_result_folder):
        os.mkdir(output_result_folder)

    encodings_names=os.listdir(encodings_folder)
    instances_names=os.listdir(instances_folder)

    for enc in encodings_names:
        run_instances_for_enc(enc,encodings_folder,instances_names,instances_folder,output_result_folder,t_cutoff)

    #analysis_result()
    #time increase according to percentage solved


    #combine results
    combine_result(output_result_folder,data_final)
