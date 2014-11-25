#include "stdafx.h"
#pragma once

class LDA
{
public:
	LDA(void);
	LDA(const std::string &file_bow, const std::string &file_vocabulary);
	virtual ~LDA(void);
	virtual void train(const int &iter);
	std::vector<std::vector<double>> calc_phi(void);
	std::vector<std::vector<double>> calc_theta(void);
	double calc_perplexity(void);
	void save_phi(const std::string &file_phi, int W_top = 0);
	void save_theta(const std::string &file_theta, int K_top = 0);
	virtual void save_model(void);

	int M; // number of documents
	int N; // number of tokens in the corpus
	int K; // number of topics
	int W; // size of vocabulary
	double ALPHA;
	double BETA;
	std::vector<std::string> vocabulary;
	std::vector<std::vector<int>> x;
	std::vector<std::vector<int>> z;
	std::vector<int> n_k;
	std::vector<std::vector<int>> n_jk;
	std::vector<std::vector<int>> n_wk;
	boost::mt19937 rng;
};

