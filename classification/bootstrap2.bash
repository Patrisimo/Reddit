
# $1 is corpus
# $2 is language name
# $3 is number of runs to do
# $4 is whether to prune
# $5 is the basename of the initial run
# $6 is tokenizer
# $7 is max_ngram

basename=$5
tokenizer=$6
round=0
while [[ $round -lt $3 ]]
do
	round=$(($round + 1))
	read -p "Which suffix? " best
	echo "Round $round"
	# Round1
	python doitall.py pick $1 ${basename}_${best}.model.gz $tokenizer 5000 ${2}_round$round --already-seen ${basename}.seen.json
	python doitall.py analyze ${2}_round$round.tsv.gz
	python doitall.py analyze ${2}_round$round.seen.tsv.gz

	basename=${2}_round$round
	if [[ $4 ]]
	then
		python doitall.py prune ${basename}.seen.tsv.gz 0.33 ${basename}_pruned
		fname=${basename}_pruned.tsv.gz
	else
		fname=$basename.seen.tsv.gz
	fi
	read -p "How many documents? " docs

	python doitall.py train_classifier $fname $docs naive_bayes $tokenizer ${basename}_nb1 --max_ngram $7
	echo "^ nb1"
	python doitall.py train_classifier $fname $docs naive_bayes $tokenizer ${basename}_nb2 --max_ngram $7
	echo "^ nb2"
	python doitall.py train_classifier $fname $docs naive_bayes $tokenizer ${basename}_nb3 --max_ngram $7
	echo "^ nb3"

	python doitall.py train_classifier $fname $docs logistic_regression $tokenizer ${basename}_lr1 --max_ngram $7
	echo "^ lr1"
	python doitall.py train_classifier $fname $docs logistic_regression $tokenizer ${basename}_lr2 --max_ngram $7
	echo "^ lr2"
	python doitall.py train_classifier $fname $docs logistic_regression $tokenizer ${basename}_lr3 --max_ngram $7
	echo "^ lr3"
	
	python doitall.py train_classifier $fname $docs svm $tokenizer ${basename}_svm1 --max_ngram $7
	echo "^ svm1"
	python doitall.py train_classifier $fname $docs svm $tokenizer ${basename}_svm2 --max_ngram $7
	echo "^ svm2"
	python doitall.py train_classifier $fname $docs svm $tokenizer ${basename}_svm3 --max_ngram $7
	echo "^ svm3"
done