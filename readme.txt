Encoding selection platform

The encoding selection platform is an automated tool to generate, evaluate, 
and select encodings. Users can only provide instance set and enocodings, 
the system will rewrite and generate new encodings, 
evaluate the performance of all encodings to generate encoding candidates, 
and finally build machine learning models to predict the best encoding 
or encoding schedule on a per-instance basis.

The system involves encoding rewriting, performance data collection, 
encoding candidate generation, instance feature extraction,
machine learning modeling, schedule generation, solution evaluation and solving.


System:
Linux or MacOS (feature extractor runs on Linux or MacOS)

Python Version: python 3.7

(with anaconda installed: run)
conda create -n platformpy3 python=3.7
conda activate platformpy3
conda deactivate
Packages used:
clingo (5.4)
numpy
pandas
sklearn

(with anaconda installed: run)
conda install -c potassco clingo=5.4 or conda install -c conda-forge clingo=5.4
conda install numpy
conda install pandas
conda install -c anaconda scikit-learn

after installation, close and reopen terminal


How to use:

python train.py
--encodings encodings_folder (default:encodings)
--instances instances_folder (default:instances)
--cutoff xxx (default:200s)
--num_candidate n (default: min(4, encodings))

python solve.py 
--new_instances new_instances_folder (default:new_instances)




Run each step:
1.Encoding rewrite

python train.py –p 1
--encodings encodings_folder (default:encodings)
--rewrite_form 0 1 2 3 (default: first available)

default input:encodings
default output:encodings

ls encodings
python train.py –p 1
ls encodings

2.Performance data generation

python train.py -p 2 
--instances instances_folder (default:instances)
--time_out xxx (default:200s)

default input:encodings, instances
default output:
--performance: performance combination of all encodings to one file
--performance_each_enc：performance combination of all instances for each encoding
--performance_each_enc_tmp：output of all instances for each encoding

ls encodings
ls instances
#python train.py -p 2
cat performance/

3.Encoding candidate generation

python train.py -p 3
--performance_data performance_folder (default: performance)
--num_candidate n (default: min(4, encodings))

default input: performance, encodings
default output: 
--performance_selected: selected performance
--performance_output: performance analysis
--encodings_selected: selected encodings


cat performance/
ls encodings
python train.py -p 3 --num_candidate 4
cat performance_selected/
ls encodings_selected

python train.py -p 3 --num_candidate 3
cat performance_selected/
ls encodings_selected


4.Feature extraction
python train.py -p 4

default input: encodings_selected, instances, performance_selected
default output: features

#python train.py -p 4
cat features/

5.Feature selection
python train.py -p 5

default input: features, performance_selected
default output: features_selected


ls features/
cat features_domain/
cat performance_selected/
python train.py -p 5
cat features_selected/


6.Model building
python train.py -p 6

default input: features_selected, performance_selected
default output: ml_hyper,ml_models

cat features_selected/
cat performance_selected/
python train.py -p 6



7.Schedule building
python train.py -p 7

default input: performance_selected
default output: schedule


8.Interleaving schedule building
python train.py -p 8

default input: performance_selected
default output: interleave


9.Solution estimation
python train.py -p 9

default input: interleave, schedule, ml_models, test_data
default output: solution

10.New instance solving
python solve.py
--new_instances new_instances_folder (default:new_instances)

default input: new_instances, encodings_selected, solution, interleave, schedule, ml_models
default output: start solving process


