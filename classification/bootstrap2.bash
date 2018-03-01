
# $1 is corpus
# $2 is language name
# $3 is number of runs to do
# $4 is whether to prune
# $5 is the basename of the initial run

basename=$5
round=0
while [[ $round -lt $3 ]]
do
	round=$(($round + 1))
	read -p "Which suffix? " best
	echo "Round $round"
	# Round1
	python doitall.py pick $1 ${basename}_${best}.model.gz bow 5000 ${2}_round$round --already-seen ${basename}.seen.json
	python doitall.py analyze ${2}_round$round.tsv.gz
	python doitall.py analyze ${2}_round$round.seen.tsv.gz

	basename=${2}_round$round
	if [[ -n $4 ]]
	then
		python doitall.py prune ${basename}.seen.tsv.gz 0.33 ${basename}_pruned
		fname=${basename}_pruned.tsv.gz
	else
		fname=$basename.seen.tsv.gz
	fi
	read -p "How many documents? " docs

	python doitall.py train_classifier $fname $docs naive_bayes bow ${basename}_nb1
	echo "^ nb1"
	python doitall.py train_classifier $fname $docs naive_bayes bow ${basename}_nb2
	echo "^ nb2"
	python doitall.py train_classifier $fname $docs naive_bayes bow ${basename}_nb3
	echo "^ nb3"

	python doitall.py train_classifier $fname $docs logistic_regression bow ${basename}_lr1
	echo "^ lr1"
	python doitall.py train_classifier $fname $docs logistic_regression bow ${basename}_lr2
	echo "^ lr2"
	python doitall.py train_classifier $fname $docs logistic_regression bow ${basename}_lr3
	echo "^ lr3"
	
	python doitall.py train_classifier $fname $docs svm bow ${basename}_svm1
	echo "^ svm1"
	python doitall.py train_classifier $fname $docs svm bow ${basename}_svm2
	echo "^ svm2"
	python doitall.py train_classifier $fname $docs svm bow ${basename}_svm3
	echo "^ svm3"
done