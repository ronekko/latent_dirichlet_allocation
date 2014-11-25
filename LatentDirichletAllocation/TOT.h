#pragma once
#include "stdafx.h"
#include "lda.h"
class TOT :
	public LDA
{
private:
	TOT(void);
public:
	TOT(const std::string &file_bow, const std::string &file_vocabulary, const std::string &file_timestamp);
	~TOT(void);
	void train(const int &iter);
	void save_psi(const std::string &file_psi);
	void save_model(void);
	
	std::vector<std::vector<double>> t; // timestamp of token[j][i]: [0, 1) normalized
	std::vector<util::BetaDistribution> psi; // psi[k]: �g�s�b�NK�̃x�[�^���z�̃p�����^��, Beta(t; psi[k].first, psi[k].second)
	std::vector<std::vector<std::vector<double>>> beta_log_likelihood; // t[j][i]���Ƃ̃x�[�^���z�ޓx�̍��̃L���b�V�� 
	// �x�[�^���z�����overfitting����̂��߂̉��z�T���v�� 
	// ���ۂ�t�ɉ�����[0,1]��T�ɓ������������̂�p����psi�𐄒肷�� 
	// T���傫���قǃX���[�W���O������������ 
	int T;
	std::vector<double> t_virtual; 
};

