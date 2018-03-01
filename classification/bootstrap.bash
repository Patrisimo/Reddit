
# $1 is corpus
# $2 is language name
# $3 is number of runs to do
# $4 is whether to prune
python doitall.py pick_random $1 5000 ${2}_random_init
python doitall.py analyze ${2}_random_init.tsv.gz
#python doitall.py analyze ${2}_random_init.seen.tsv.gz

if [[ -n $4 ]]
then
	python doitall.py prune ${2}_random_init.tsv.gz 0.33 ${2}_random_init_pruned
fi
read -p "How many documents? " docs

basename=${2}_random_init
if [[ -n $4 ]]
then
	fname=${2}_random_init_pruned.tsv.gz
else
	fname=${2}_random_init.tsv.gz
fi

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

bash bootstrap2.bash $1 $2 $3 "$4" $basename
