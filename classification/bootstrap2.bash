
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

	python compare.py $fname $docs $basename
	
done