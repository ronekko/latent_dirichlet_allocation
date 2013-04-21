#include "stdafx.h"
#include "LDA.h"

using namespace std;

LDA::LDA(void)
{
}


LDA::~LDA(void)
{
}


LDA::LDA(const string &file_bow, const string &file_vocabulary)
{
	rgen.seed(0);
	K = 50;

	// コーパスのカウントファイルのロード
	ifstream ifs_bow(file_bow);
	string line;
	M = 0;
	while(getline(ifs_bow, line))
	{
		using namespace boost::algorithm;
		vector<string> tokens;
		
		split(tokens, line, is_space());
		
		int N = tokens.size();
		vector<int> words(N);
		vector<int> counts(N, 0);

		for(int n=0; n<N; ++n){
			sscanf_s(tokens[n].c_str(), "%d:%d", &(words[n]), &(counts[n]));
		}

		int N_j = boost::accumulate(counts, 0);
		vector<int> x_j(N_j);
		for(int n=0, i=0; n<N; ++n){
			int word = words[n];
			for(int c=0; c<counts[n]; ++c){
				x_j[i] = word;
				++i;
			}
		}
		boost::random_shuffle(x_j); // 文書内のトークンの出現順をシャッフルする
		x.push_back(x_j);
		++M;
	}
	
	// コーパスの語彙のロード
	ifstream ifs_vocabulary(file_vocabulary);
	W = 0;
	while(getline(ifs_vocabulary, line))
	{
		boost::algorithm::trim(line);
		vocabulary.push_back(line);
		++W;
	}
	
	// zの初期化、カウントの初期化
	z = vector<vector<int>>(M);
	n_k = vector<int>(K, 0);
	n_kw = vector<vector<int>>(K, vector<int>(W, 0));
	n_jk = vector<vector<int>>(M, vector<int>(K, 0));
	boost::uniform_int<> uint(0, K-1);
	for(int j=0; j<M; ++j){
		int N_j = x[j].size();
		z[j] = vector<int>(N_j);
		for(int i=0; i<N_j; ++i){
			int k = uint(rgen);
			z[j][i] = k;

			int w = x[j][i];
			n_k[k]++;
			n_kw[k][w]++;
			n_jk[j][k]++;
		}
	}
}
