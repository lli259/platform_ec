import argparse,os
import numpy as np
import pandas as pd



def define_args(arg_parser):

    arg_parser.add_argument('--performance_folder', nargs='*', default=['performance_selected'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')
    arg_parser.add_argument('--schedule_output', nargs='*', default=['schedule'], help='Gringo input files') 

def seq_two_time(t1,t2,t_given1,t_given2):
    if t1<t_given1:
        return t1
    else:
        if t2<t_given2:
            return t_given1+t2
        else:
            return t_given1+t_given2



def get_seq_diff_time(df,col_name1,col_name_2,t_given1,t_given2,cutoff):
    insts=df.index.values
    runtime_list=[]
    for ins_id in insts:
        t1=df.loc[ins_id,col_name1]
        t2=df.loc[ins_id,col_name_2]
        sq_t=seq_two_time(t1,t2,t_given1,t_given2)
        runtime_list.append(sq_t)
    solved=0
    total_time=0
    for rt in runtime_list:
        if rt<cutoff:
            solved+=1
        total_time+=rt
    leng=float(len(runtime_list))
    return solved/leng,total_time/leng

def build(cutoff,df_file):

    cutoff=int(cutoff)
    df=pd.read_csv(df_file)
    df=df.set_index(df.columns[0])
    cols=df.columns.values

    
    alltime_best=[]
    for t1 in range(0,cutoff+1,1):
        t_given1=t1
        t_given2=cutoff-t1
        allresult=[]
        for i in range(0,len(cols)):
            for j in range(i,len(cols)):
                col_name1=cols[i]
                col_name_2=cols[j]
                if col_name1!=col_name_2:
                    s,t=get_seq_diff_time(df,col_name1,col_name_2,t_given1,t_given2,cutoff)
                    allresult.append((s,t,col_name1,col_name_2))
                #print(col_name1,'+',col_name_2,",",s,",",t)
        #print("\n")
        #print(allresult)
        allresult2=sorted(allresult,key=lambda v:v[0])
        #print(allresult2)
        best=allresult2[-1]
        s,t,col_name1,col_name_2=best
        alltime_best.append((s,t,col_name1,col_name_2,t_given1,t_given2))
    alltime_best=sorted(alltime_best,key=lambda v:v[0])
    best=alltime_best[-1]
    s,t,col_name1,col_name_2,t_given1,t_given2=best
    #print('best solving','%:',s,'time:',t)
    print('schedule',col_name1,col_name_2,t_given1,t_given2)

    with open('evaluation/result.csv','a') as f:
        f.write('schedule,'+str(round(s,2))+'\n')

    return col_name1,col_name_2,t_given1,t_given2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()

    performance_folder=args.performance_folder[0]
    cutoff=args.cutoff[0]
    schedule_out_folder=args.schedule_output[0]

    df_file=os.listdir(performance_folder)[0]
    df_file=performance_folder+'/'+df_file

    col_name1,col_name_2,t_given1,t_given2=build(cutoff,df_file)

    if not os.path.exists(schedule_out_folder):
        os.system('mkdir '+schedule_out_folder)
    
    with open(schedule_out_folder+'/'+'schedule.csv','w') as f:
        f.write('enc1,enc2,t_given1,t_given2\n')
        f.write(str(col_name1)+','+str(col_name_2)+','+str(t_given1)+','+str(t_given2)+'\n')




