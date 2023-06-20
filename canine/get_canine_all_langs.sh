
for LANG in bg cs da de en eo es eu fi hi hu hy ja ko la lt mk nl ro ru tl uk vi
do
  echo Getting embeddings for language $LANG
  python3 get_canine_embeddings_general.py $LANG
done
