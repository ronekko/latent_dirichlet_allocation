// LatentDirichletAllocation : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "LDA.h"

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	const string file_bow = "NIPS0-12/counts0-12.txt";
	const string file_vocabulary = "NIPS0-12/voca0-12.txt";
	LDA::LDA(file_bow, file_vocabulary);

	return 0;
}

