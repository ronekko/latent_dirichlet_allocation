#include "stdafx.h"
#pragma once

using namespace std;
class LDA
{
public:
	LDA(void);
	LDA(const string &file_bow, const string &file_vocabulary);
	~LDA(void);

	int M; // number of documents
	int K; // number of topics
	int W; // size of vocabulary
	double alpha;
	double beta;
	vector<string> vocabulary;
	vector<vector<int>> x;
	vector<vector<int>> z;
	vector<int> n_k;
	vector<vector<int>> n_jk;
	vector<vector<int>> n_kw;
	mt19937 rgen;
};

