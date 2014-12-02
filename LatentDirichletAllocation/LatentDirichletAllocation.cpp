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

	LDA lda(60, 0, 1.0, 20.0);
	//TOT lda(file_bow, file_vocabulary, file_timestamp, 60, 0);

	vector<std::unordered_map<int, int>> bows = LDA::load_bow_file(file_bow);
	lda.fit(bows);
	
	lda.save_model("result_phi.txt", "result_theta.txt", 50, 10);

	return 0;
}

