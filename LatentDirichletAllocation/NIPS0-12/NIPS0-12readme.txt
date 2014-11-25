ID、ラベルは全て0始まり（1始まりではない）
counts0-12.txt: 1行で1文書、[wordtype:count ]*の形式。文書IDは0始まり。
pclass0-12.txt: 1行で1文書、0～12のラベルでで1987～1999年を表す
voca0-12.txt  : 語彙。最初の単語(network)が単語ラベル0に対応。
nips0-12for_my_lda_implementation.txt: countsのフォーマットを自作のLDAに入力できる形式に合わせたもの（多分0始まりかどうかを合わす必要がある）