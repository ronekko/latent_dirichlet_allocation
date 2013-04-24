#pragma once
#include "lda.h"
class TOT :
	public LDA
{
public:
	TOT(void);
	TOT(const string &file_bow, const string &file_vocabulary, const string &file_timestamp);
	~TOT(void);
	void train(const int &iter);
	void save_psi(const string &file_psi);
	void save_model(void);
	
	vector<vector<double>> t; // timestamp of token[j][i]: [0, 1) normalized
	vector<pair<double, double>> psi; // psi[k]: �g�s�b�NK�̃x�[�^���z�̃p�����^��, Beta(t; psi[k].first, psi[k].second)
	vector<vector<double>> beta_likelihood; // t[j][i]���Ƃ̃x�[�^���z�ޓx�̍��̃L���b�V�� 
};

