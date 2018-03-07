
if [[ -n $2 ]]
then
	type=$2
else
	type=naive_bayes
fi
if [[ -n $3 ]]
then
	token=$3
else
	token=bow
fi
	

python patrick_algs.py --input $1_train.tsv --type $type --train TRAIN --tokens $token --output $1_$2.model.gz
python patrick_algs.py --input $1_test.tsv --test TEST --tokens $token --output $1_$2.prob.gz --model $1_$2.model.gz
python evaluate.py $1_$2.prob.gz