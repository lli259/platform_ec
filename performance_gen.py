import argparse,os
import numpy as np
from collections import Counter
import pandas as pd

def define_args(arg_parser):

    arg_parser.add_argument('--encodings', nargs='*', default=['encodings'], help='Gringo input files')
    arg_parser.add_argument('--instances', nargs='*', default=['instances'], help='Gringo input files')    
    arg_parser.add_argument('--performance_data', nargs='*', default=['performance'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')



def run_instances_for_enc(enc_name,encodings_folder,instances_names,instances_folder,out_folder,cutoff_t):

    #create folder for each encoding to store result
    enc_folder=out_folder+'/'+enc_name
    if not os.path.exists(enc_folder):
        os.system('mkdir '+enc_folder)
    
    print('solving instances using '+enc_name)

    for instances in instances_names:  
        resultname=enc_name+'_'+instances
        os.system('./tools/gringo '+encodings_folder+'/'+enc_name
        +' '+instances_folder+'/'+enc_name
        +' | ./tools/clasp > '
        + enc_folder+'/'+resultname)

def get_perform_result(perf_temp_folder,enc_folder,result_f):
    #input performance file
    #output: instance,model,time
    f=perf_temp_folder+'/'+enc_folder+'/'+result_f
    instance=result_f
    model,time='',''
    with open(f,'r') as fopen:
        lines=fopen.readlines()
    for line in lines:
        if 'model' in line.lower():
            model=line.lower()[-3:]
        if 'CPU' in line.lower():
            time=line.lower()[-3:]   
    return instance,model,time

def combine_enc_and_save(perf_temp_folder,enc_folder):
    allfiles=os.listdir(perf_temp_folder+'/'+enc_folder)
    savefile=enc_folder+'.csv'
    data_store_folder=output_temp[:-4]
    allline=[]
    allline.append('inst,model,time\n')
    for result_f in allfiles:
        inst,model,time=get_perform_result(perf_temp_folder,enc_folder,result_f)
        allline.append(','.join([inst,model,time])+'\n')
    with open(data_store_folder+'/'+savefile,'r') as fopen:
        for line in allline:
            fopen.write(line)


def combine_result(data_folder1,data_folder2):
    output_file='performance.csv'
    for enc in os.listdir(data_folder1):
        combine_enc_and_save(data_folder1,enc)

    data_store_folder=data_folder1[:-4]
    allcsv=os.listdir(data_store_folder)

    pd1=pd.read_csv(allcsv[0])
    cols=pd1.columns.names
    pd1=pd1.set_index(cols[0])
    cols=[ col+'_'+allcsv[0] for col in cols[1:]]
    pd1.columns.names=cols

    for pointer in (1,len(allcsv)):
        pd11=pd.read_csv(allcsv[pointer])
        cols=pd11.columns.names
        pd11=pd11.set_index(cols[0])
        cols=[ col+'_'+allcsv[0] for col in cols[1:]]
        pd11.columns.names=cols
        pd1=pd1.join(pd11)
    pd1.save_to_file(data_folder2+'/'+output_file)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()


    encodings_folder=args.encodings[0]
    instances_folder=args.instances[0]
    t_cutoff=int(args.cutoff[0])
    data_final=args.performance_data[0]
    output_temp=data_final+'_each_enc_tmp'
    output_result_folder=output_temp[:-4]

    if not os.path.exists(output_temp):
        os.system('mkdir '+output_temp)

    if not os.path.exists(output_result_folder):
        os.system('mkdir '+output_result_folder)

    encodings_names=os.listdir(encodings_folder)
    instances_names=os.listdir(instances_folder)

    for enc in encodings_names:
        run_instances_for_enc(enc,encodings_folder,instances_names,instances_folder,output_temp,t_cutoff)

    #analysis_result()
    #time increase according to percentage solved


    #combine results
    combine_result(output_temp,data_final)
