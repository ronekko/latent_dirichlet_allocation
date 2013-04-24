#include "stdafx.h"
#pragma once

using namespace std;
class LDA
{
public:
	LDA(void);
	LDA(const string &file_bow, const string &file_vocabulary);
	virtual ~LDA(void);
	virtual void train(const int &iter);
	vector<vector<double>> calc_phi(void);
	vector<vector<double>> calc_theta(void);
	double calc_perplexity(void);
	void save_phi(const string &file_phi, int W_top=0);
	void save_theta(const string &file_theta, int K_top=0);
	virtual void save_model(void);

	int M; // number of documents
	int N; // number of tokens in the corpus
	int K; // number of topics
	int W; // size of vocabulary
	double ALPHA;
	double BETA;
	vector<string> vocabulary;
	vector<vector<int>> x;
	vector<vector<int>> z;
	vector<int> n_k;
	vector<vector<int>> n_jk;
	vector<vector<int>> n_wk;
	boost::mt19937 rgen;
};

