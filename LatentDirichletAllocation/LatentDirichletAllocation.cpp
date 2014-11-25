// LatentDirichletAllocation : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "LDA.h"
#include "TOT.h"

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	const string file_bow = "NIPS0-12/counts0-12.txt";
	const string file_vocabulary = "NIPS0-12/voca0-12.txt";
	const string file_timestamp = "NIPS0-12/pclass0-12.txt";

	//LDA lda(file_bow, file_vocabulary, 60, 0);
	TOT lda(file_bow, file_vocabulary, file_timestamp, 60, 0);
	
	lda.train(2000);

	return 0;
}

