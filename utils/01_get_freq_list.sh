

LANG=$1
N=5000

wget https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/freq/${LANG}.freq.gz
gunzip ${LANG}.freq.gz

cat ${LANG}.freq | sed 's/^\ *//g' | cut -f2 -d' ' | ggrep -P '[\p{Latin}]' | head -n$N > ${LANG}.mostFreq$N

#ggrep -P '[\p{Hangul}]'
#\p{Latin}
#\p{Devanagari}
#\p{Armenian} 
#\p{Cyrillic}
#\p{Hiragana}
#\p{Katakana}

mv ${LANG}.mostFreq$N ../canine/frequency_lists
