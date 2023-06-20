
#Get narrow pronunciations from
#https://github.com/CUNY-CL/wikipron/tree/master/data/scrape/tsv
#rm engwords.txt engprons.txt engwords_sep.txt eng_wikipron_news_input.tsv eng_wikipron_news_input.tsv eng_wikipron_news_output.tsv eng_wikipron_news_output.tsv eng_wikipron_aligned.tsv

WIKIPRONFILE=$1
LANG=$2

cat $WIKIPRONFILE | cut -f1 > tmp.${WIKIPRONFILE}-words
cat $WIKIPRONFILE | cut -f2 > tmp.${WIKIPRONFILE}-prons
cat tmp.${WIKIPRONFILE}-words | sed 's/./& /g;s/\ $//' >  tmp.${WIKIPRONFILE}-wordssep
paste tmp.${WIKIPRONFILE}-wordssep tmp.${WIKIPRONFILE}-prons > tmp.${WIKIPRONFILE}-newsform.input

cd ../../m2m-aligner/
./m2m-aligner -i ../contextual_char_emb2022/utils/tmp.${WIKIPRONFILE}-newsform.input -o  ../contextual_char_emb2022/utils/tmp.${WIKIPRONFILE}-newsform.output

cd ../contextual_char_emb2022/utils

cat tmp.${WIKIPRONFILE}-newsform.output | cut -f1 | sed 's/://g;s/\|//g' > tmp.${WIKIPRONFILE}-newsform.output_words
cat tmp.${WIKIPRONFILE}-newsform.output | sed 's/://g;s/\|/ /g' > tmp.${WIKIPRONFILE}-newsform.alignedcols12

paste tmp.${WIKIPRONFILE}-newsform.alignedcols12 tmp.${WIKIPRONFILE}-newsform.output_words | sort > ${WIKIPRONFILE}.aligned

mv ${WIKIPRONFILE}.aligned ../wikipron/${LANG}
mv ${WIKIPRONFILE} ../wikipron/
