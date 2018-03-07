for lang in spanish french italian
do
	for var in "-sub" ""
	do
		for feat in contro gilded score
		do
			for alg in naive_bayes logistic_regression
			do
				echo "Evaluating $alg on $lang${var}_$feat"
				python patrick_algs.py --input $lang${var}_${feat}_train.tsv --type $alg --train TRAIN --tokens bow --output $lang${var}_${feat}_$alg.model.gz
				python patrick_algs.py --input $lang${var}_${feat}_test.tsv --test TEST --tokens bow --output $lang${var}_${feat}_$alg.prob.gz --model $lang${var}_${feat}_$alg.model.gz
				python evaluate.py $lang${var}_${feat}_$alg.prob.gz
			done
		done	
	done
done
