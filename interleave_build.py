import argparse,os
import numpy as np
import pandas as pd


def define_args(arg_parser):

    arg_parser.add_argument('--performance_folder', nargs='*', default=['performance_selected'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')
    arg_parser.add_argument('--interleave_out', nargs='*', default=['interleave'], help='Gringo input files') 



class interleave_class():
    def __init__(self):
        pass
    #input list, cap_runtime
    #return solve percentage, avg time
    def solve_perc_avg_time(self,input_list,cap_runtime):
        leng=len(input_list)*1.0
        sov=sum([i<cap_runtime for i in input_list])
        time=sum(input_list)
        return sov/leng,time/leng


    #input: list of runtime, time_each,total_time
    #for example [10,3,13], 4, 20
    #1,2,3,1,2,3,...
    #return runtime of interleave_runing
    def interleave_runing(self,list_solver_time,time_each,total_time):
        timeout=False
        index=0
        used_time_list=[0 for i in list_solver_time]

        #init: all time set 0, start from index 0
        #continue solving, until one's time < time_each+ his used time
        #then he uses his time in total. other use the used time.
        while not timeout:
            if list_solver_time[index]<=time_each+used_time_list[index]:
                solved=index 
                break
            else:
                used_time_list[index]+=time_each
                index+=1
                if index==len(list_solver_time):
                    index=0

            if sum(used_time_list)>=total_time:
                timeout=True
        
        if timeout:
            return total_time

        re=0
        for i in range(len(list_solver_time)):
            if i==solved:
                re+=list_solver_time[i]
            else:
                re+=used_time_list[i]
        return re


    def interleave_run_four_lists(self,la,lb,lc,ld,time_each=20,total_time=400):
        re=[]
        for i in range(len(la)):
            list_time=[la[i],lb[i],lc[i],ld[i]]
            r=self.interleave_runing(list_time,time_each,total_time)
            re.append(r)
        return re

    def interleave_run_three_lists(self,la,lb,lc,time_each=20,total_time=400):
        re=[]
        for i in range(len(la)):
            list_time=[la[i],lb[i],lc[i]]
            r=self.interleave_runing(list_time,time_each,total_time)
            re.append(r)
        return re

    def interleave_run_two_lists(self,la,lb,time_each=20,total_time=400):
        re=[]
        for i in range(len(la)):
            list_time=[la[i],lb[i]]
            r=self.interleave_runing(list_time,time_each,total_time)
            re.append(r)
        return re

    #input 4 lists, time range, time each
    def interleave_diff_tm_n_ord_4list_(self,input_la,input_lb,input_lc,input_ld,rs,re,r,total_time=400):
        #print(column)
        allbest=[]
        alldata=[input_la,input_lb,input_lc,input_ld]
        for time_each in range(rs,re,r):
            #print(rs,re,r,time_each)
            allfour=[]
            size=4
            for i in range(size):
                for j in range(size):
                    for a in range(size):
                        for b in range(size):
                            if not (i==j or i==a or i==b or j==a or j==b or a==b):
                                la,lb,lc,ld=alldata[i],alldata[j],alldata[a],alldata[b]
                                re_list=self.interleave_run_four_lists(la,lb,lc,ld,time_each,total_time)
                                s,t=self.solve_perc_avg_time(re_list,total_time)
                                si,sj,sa,sb=str(i),str(j),str(a),str(b)
                                #print(si,sj,sa,sb,s,t)
                                allfour.append((s,t,'-'.join([si,sj,sa,sb])))
            allfour=sorted(allfour)
            #time, best result and order in the time
            s,t,sched=allfour[-1]
            allbest.append((s,t,sched,time_each))
        allbest=sorted(allbest)
        return allbest[-1]



    def interleave_diff_tm_n_ord_3list_(self,input_la,input_lb,input_lc,rs,re,r,total_time=400):
        #print(column)
        allbest=[]
        alldata=[input_la,input_lb,input_lc]
        for time_each in range(rs,re,r):
            allthree=[]
            size=3
            for i in range(size):
                for j in range(size):
                    for a in range(size):
                            if not (i==j or i==a or j==a ):
                                la,lb,lc=alldata[i],alldata[j],alldata[a]

                                re_list=self.interleave_run_three_lists(la,lb,lc,time_each,total_time)
                                s,t=self.solve_perc_avg_time(re_list,total_time)

                                si,sj,sa=str(i),str(j),str(a)
                                allthree.append((s,t,'-'.join([si,sj,sa])))

            allthree=sorted(allthree)
            s,t,sched=allthree[-1]
            allbest.append((s,t,sched,time_each))
        allbest=sorted(allbest)
        return allbest[-1]


    def interleave_diff_tm_n_ord_2list_(self,input_la,input_lb,rs,re,r,total_time=400):
        #print(column)
        allbest=[]
        alldata=[input_la,input_lb]
        for time_each in range(rs,re,r):
            allret=[]
            size=2
            for i in range(size):
                for j in range(size):
                            if not (i==j):
                                la,lb=alldata[i],alldata[j]
                                re_list=self.interleave_run_two_lists(la,lb,time_each,total_time)
                                s,t=self.solve_perc_avg_time(re_list,total_time)
                                si,sj=str(i),str(j)
                                allret.append((s,t,'-'.join([si,sj])))

            allret=sorted(allret)
            s,t,sched=allret[-1]
            allbest.append((s,t,sched,time_each))
        allbest=sorted(allbest)
        return allbest[-1]    


def savetofile(fold,cont):
    with open(fold+'/'+'interleave.csv','w') as f:
        f.write('t,order\n')
        s,st,interl,time=cont
        #print('best solving','%:',s,'time:',st)
        print('interleaving schedule',interl,time)
        f.write(str(time)+","+str(interl))

    with open('evaluation/result.csv','a') as f:
        f.write('interleaving,'+str(round(s,2))+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()

    performance_folder=args.performance_folder[0]
    cutoff=args.cutoff[0]
    interleave_out_folder=args.interleave_out[0]

    df_file=os.listdir(performance_folder)[0]
    df_file=performance_folder+'/'+df_file

    if not os.path.exists(interleave_out_folder):
        os.system('mkdir '+interleave_out_folder)

    df=pd.read_csv(df_file)
    df=df.set_index(df.columns[0])
    column=df.columns.values

    if len(column) ==1:
        print('ERROR: there should be at least two encodings!')
        exit()

    if len(column) ==2:
        input_la,input_lb=df[column[0]],df[column[1]]
        intlv=interleave_class()
        best=intlv.interleave_diff_tm_n_ord_2list_(input_la,input_lb,1,int(int(cutoff)/len(column)),1,total_time=int(cutoff))
        s,st,interl,time=best
        interl=interl.split('-')
        interl=[column[int(i)] for i in interl]
        interl='-'.join(interl)
        best=s,st,interl,time
        savetofile(interleave_out_folder,best)

    else:
        if len(column) ==3:
            input_la,input_lb,input_lc=df[column[0]],df[column[1]],df[column[2]]
            intlv=interleave_class()
            best=intlv.interleave_diff_tm_n_ord_3list_(input_la,input_lb,input_lc,1,int(int(cutoff)/len(column)),1,total_time=int(cutoff))
            s,st,interl,time=best
            interl=interl.split('-')
            interl=[column[int(i)] for i in interl]
            interl='-'.join(interl)
            best=s,st,interl,time
            savetofile(interleave_out_folder,best)
        else:
            input_la,input_lb,input_lc,input_ld=df[column[0]],df[column[1]],df[column[2]],df[column[3]]

            intlv=interleave_class()
            best=intlv.interleave_diff_tm_n_ord_4list_(input_la,input_lb,input_lc,input_ld,1,int(int(cutoff)/len(column)),1,total_time=int(cutoff))
            s,st,interl,time=best
            interl=interl.split('-')
            interl=[column[int(i)] for i in interl]
            interl='-'.join(interl)
            best=s,st,interl,time
            savetofile(interleave_out_folder,best)