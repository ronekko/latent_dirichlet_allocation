// An experimental implementation of
// "Topics over time: a non-Markov continuous-time model of topical trends."
// Wang, Xuerui, and Andrew McCallum, SIGKDD, 2006.

#pragma once
#include "stdafx.h"
#include "lda.h"
class TOT :
	public LDA
{
private:
	TOT(void);
public:
	TOT(const std::string &file_bow, const std::string &file_vocabulary, const std::string &file_timestamp, const int &K, const int &seed = -1, const double &alpha_total_mass_ = 1.0, const double &beta_total_mass_ = 20.0);
	~TOT(void);
	void train(const int &iter);
	void save_psi(const std::string &file_psi);
	void save_model(void);
	
	std::vector<std::vector<double>> t; // timestamp of token[j][i]: [0, 1) normalized
	std::vector<util::BetaDistribution> psi; // psi[k]: トピックKのベータ分布のパラメタ対, Beta(t; psi[k].first, psi[k].second)
	std::vector<std::vector<std::vector<double>>> beta_log_likelihood; // t[j][i]ごとのベータ分布尤度の項のキャッシュ 
	// ベータ分布推定のoverfitting回避のための仮想サンプル 
	// 実際のtに加えて[0,1]をT個に等分割したものを用いてpsiを推定する 
	// Tが大きいほどスムージングが強くかかる 
	int T;
	std::vector<double> t_virtual; 
};

