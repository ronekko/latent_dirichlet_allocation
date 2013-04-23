#pragma once
#include "lda.h"
class TOT :
	public LDA
{
public:
	TOT(void);
	TOT(const string &file_bow, const string &file_vocabulary, const string &file_timestamp);
	~TOT(void);
	void TOT::train(const int &iter);
	
	vector<vector<double>> t; // timestamp of token[j][i]: [0, 1) normalized
	vector<pair<double, double>> psi; // psi[k]: トピックKのベータ分布のパラメタ対, Beta(t; psi[k].first, psi[k].second)
};

