// An implementation of latent Dirichlet allocation (LDA) with some learning methods.
// CGS in "Finding scientific topics.", Griffiths, Thomas L. and Mark Steyvers, PNAS 2004. 
// CVB0 in "On Smoothing and Inference for Topic Models", Asuncion, Welling, Smith and Teh, UAI 2009.

#include "stdafx.h"
#pragma once

class LDA
{
private:
	LDA(void);
public:
	enum Method{
		CGS, // collapsed Gibbs sampling
		CVB0 // collapsed variational Bayes with 0th order approximation
	};
	
	LDA(const int &K_, const double &alpha_total_mass_ = 1.0, const double &beta_total_mass_ = 20.0, int n_iter_ = 500, LDA::Method method_ = LDA::CGS, const int &seed = -1);
	virtual ~LDA(void);
	void fit(std::vector<std::unordered_map<int, int>> bows);
	std::vector<std::vector<double>> transform(std::vector<std::unordered_map<int, int>> bows);
	virtual void train_by_CGS(const int &iter);
	void train_by_CVB0(const int &iter);
	std::vector<std::vector<double>> calc_phi(void);
	std::vector<std::vector<double>> calc_theta(void);
	double calc_perplexity(void);
	void save_phi(const std::string &file_phi, int W_top = 0, const std::string &file_vicabulary = "");
	void save_theta(const std::string &file_theta, int K_top = 0);
	virtual void save_model(const std::string &file_phi = "phi.txt", const std::string &file_theta = "theta.txt", const int &W_top = 0, const int &K_top = 0, const std::string &file_vocabulary = "");
	static std::vector<std::unordered_map<int, int>> load_bow_file(const std::string &file_bow);

	int M; // number of documents
	int N; // number of tokens in the corpus
	int K; // number of topics
	int W; // size of vocabulary
	double alpha_k; // concentration parameter of symmetric Dir(\theta_j ; alphla_1,...,alpha_K)
	double beta_w;	// concentration parameter of symmetric Dir(\phi_k ; beta_1,...,beta_K)
	double alpha_total_mass; // K * alpha_k
	double beta_total_mass;  // W * beta_w 
	int n_iter;
	Method method;
	std::vector<std::vector<int>> x;	// x[j][i]: word type of i-th token in j-th document
	std::vector<std::vector<int>> z;	// z[j][i]: latent topic assignment of i-th token in j-th document
	std::vector<double> n_k;				// n_k[k] : counts of topic assignments to k-th topic
	std::vector<std::vector<double>> n_jk; // n_jk[j][k]: counts of topic assignments to k-th topic in j-th document
	std::vector<std::vector<double>> n_wk; // n_wk[w][k]: counts of word-type w's assigned to k-th topic
	boost::mt19937 rng;
};

