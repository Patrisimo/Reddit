
# $1 is corpus
# $2 is language name
# $3 is number of runs to do
# $4 is whether to prune
# $5 is tokenizer

max_ngram=1
prune=''

while getopts ":pn:" opt; do
  case $opt in
    n)
			max_ngram=$OPTARG
      ;;
		p)
			prune=1
			;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
			echo "bash $0 corpus output #runs tokenizer"
      ;;
		:)
			echo "Option -$OPTARG requires an argument" >&2
			;;
  esac
done



if [[ -z $1 ]]
then
	echo "bash $0 corpus output #runs tokenizer"
	exit 1
fi

python doitall.py pick_random $1 5000 ${2}_random_init
python doitall.py analyze ${2}_random_init.tsv.gz
#python doitall.py analyze ${2}_random_init.seen.tsv.gz

if [[ -n $4 ]]
then
	tokenizer=bow
fi

if [[ $prune ]]
then
	python doitall.py prune ${2}_random_init.tsv.gz 0.33 ${2}_random_init_pruned
fi
read -p "How many documents? " docs

basename=${2}_random_init
if [[ $prune ]]
then
	fname=${2}_random_init_pruned.tsv.gz
else
	fname=${2}_random_init.tsv.gz
fi

python doitall.py train_classifier $fname $docs naive_bayes $tokenizer ${basename}_nb1 --max_ngram $max_ngram
echo "^ nb1"
python doitall.py train_classifier $fname $docs naive_bayes $tokenizer ${basename}_nb2 --max_ngram $max_ngram
echo "^ nb2"
python doitall.py train_classifier $fname $docs naive_bayes $tokenizer ${basename}_nb3 --max_ngram $max_ngram
echo "^ nb3"

python doitall.py train_classifier $fname $docs logistic_regression $tokenizer ${basename}_lr1 --max_ngram $max_ngram
echo "^ lr1"
python doitall.py train_classifier $fname $docs logistic_regression $tokenizer ${basename}_lr2 --max_ngram $max_ngram
echo "^ lr2"
python doitall.py train_classifier $fname $docs logistic_regression $tokenizer ${basename}_lr3 --max_ngram $max_ngram
echo "^ lr3"

python doitall.py train_classifier $fname $docs svm $tokenizer ${basename}_svm1 --max_ngram $max_ngram
echo "^ svm1"
python doitall.py train_classifier $fname $docs svm $tokenizer ${basename}_svm2 --max_ngram $max_ngram
echo "^ svm2"
python doitall.py train_classifier $fname $docs svm $tokenizer ${basename}_svm3 --max_ngram $max_ngram
echo "^ svm3"

bash bootstrap2.bash $1 $2 $3 "$prune" $basename $tokenizer $max_ngram
