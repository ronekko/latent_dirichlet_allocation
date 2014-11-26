#include "stdafx.h"
#include "TOT.h"
#include "utility.hpp"
#include <omp.h>

using namespace std;



TOT::~TOT(void)
{
}

TOT::TOT(const string &file_bow, const string &file_vocabulary, const string &file_timestamp, const int &K, const int &seed, const double &alpha_total_mass_, const double &beta_total_mass_)
	: LDA(file_bow, file_vocabulary, K, seed, alpha_total_mass_, beta_total_mass_)
{
	T = 10000;
	// �R�[�p�X�̃^�C���X�^���v�t�@�C���̃��[�h
	vector<double> timestamps;
	timestamps.reserve(M);
	ifstream ifs_timestamp(file_timestamp);
	string line;
	while(getline(ifs_timestamp, line))
	{
		double timestamp = boost::lexical_cast<double>(line);
		timestamps.push_back(timestamp);
	}
	
	double min_timestamp = *boost::min_element(timestamps);
	double max_timestamp = *boost::max_element(timestamps);
	// ���Ƃ��΁A�^�C���X�^���v��2000�`2010�Ȃ�11�N�Ԃ���̂�max_timestamp+1����
	double range_timestamp = max_timestamp - min_timestamp + 1.0;

	t = vector<vector<double>>(M);
	for(int j=0; j<M; ++j){
		int N_j = x[j].size();
		t[j] = vector<double>(N_j);
		for(int i=0; i<N_j; ++i){
			double timestamp = timestamps[j];
			// �����̃^�C���X�^���v�ɑΉ������Ԃ̒����̒l�i��N�Ԃ̒����j�𕶏��̃^�C���X�^���v�Ƃ���
			double normalized_timestamp = (timestamp - min_timestamp + 0.5) / range_timestamp;
			t[j][i] = normalized_timestamp;
		}
	}

	// �p�����^�A�ޓx�̃L���b�V���̏�����
	psi = vector<util::BetaDistribution>(K, util::BetaDistribution(1.0, 1.0));
	beta_log_likelihood = vector<vector<vector<double>>>(M);
#pragma omp parallel for
	for(int j=0; j<M; ++j){
		int N_j = t[j].size();
		beta_log_likelihood[j] = vector<vector<double>>(N_j);
		for(int i=0; i<N_j; ++i){
			beta_log_likelihood[j][i] = vector<double>(K);
			for(int k=0; k<K; ++k){
				beta_log_likelihood[j][i][k] = psi[k].log_pdf(t[j][i]);
			}
		}
	}

	t_virtual = vector<double>(T);
	for(int i=0; i<T; ++i){
		t_virtual[i] = i / static_cast<double>(T - 1);
	}
}


void TOT::train(const int &iter)
{
	int interval_beta_update = 1;
	for(int r=0; r<iter; ++r)
	{
		boost::timer timer;

		for(int j=0; j<M; ++j)
		{
			int N = x[j].size();
			for(int i=0; i<N; ++i)
			{
				int w = x[j][i];
				int k_old = z[j][i];

				n_k[k_old]--;
				n_wk[w][k_old]--;
				n_jk[j][k_old]--;

				vector<double> p(K);
				for(int k=0; k<K; ++k){
					p[k] = log((n_jk[j][k] + alpha_k) * (n_wk[w][k] + beta_w) / (n_k[k] + W * beta_w)) + beta_log_likelihood[j][i][k];
				}
				int k_new = util::multinomialByUnnormalizedLogParameters(rng, p);
				
				z[j][i] = k_new;

				n_k[k_new]++;
				n_wk[w][k_new]++;
				n_jk[j][k_new]++;
			}
		}
		
		// �x�[�^���z�p�����^�Ɩޓx�̍X�V�iiter��interval_beta_update�񂲂ƂɁj
		if((r % interval_beta_update) == (interval_beta_update - 1)){
			vector<double> t_mean(K, 0.0);
			vector<double> t_var(K, 0.0);
			for(int j=0; j<M; ++j){
				for(int i=0; i<t[j].size(); ++i){
					int k = z[j][i];
					t_mean[k] += t[j][i];
				}
			}
			for(int k=0; k<K; ++k){
				// �X���[�W���O�� 
				t_mean[k] += boost::accumulate(t_virtual, 0.0);
				t_mean[k] /= static_cast<double>(n_k[k] + T);
			}
			for(int j=0; j<M; ++j){
				for(int i=0; i<t[j].size(); ++i){
					int k = z[j][i];
					t_var[k] += (t[j][i] - t_mean[k]) * (t[j][i] - t_mean[k]);
				}
			}
			for(int k=0; k<K; ++k){
				// �X���[�W���O��
				for(auto &tv : t_virtual){
					t_var[k] += (tv - t_mean[k]) * (tv - t_mean[k]);
				}
				t_var[k] /= static_cast<double>(n_k[k] + T);
				// TOT�_����Appendix�̍Ō�A�Ղ����߂鎮�ӂ��̑�2���i���ʂ̒��j
				double last_term = t_mean[k] * (1.0 - t_mean[k]) / t_var[k] - 1.0;
				double alpha = t_mean[k] * last_term;
				double beta = (1.0 - t_mean[k]) * last_term;
				psi[k] = util::BetaDistribution(alpha, beta);
			}
		
			// �x�[�^���z�ޓx���̍X�V
			#pragma omp parallel for
			for(int j=0; j<M; ++j){
				for(int i=0; i<t[j].size(); ++i){
					for(int k=0; k<K; ++k){
						beta_log_likelihood[j][i][k] = psi[k].log_pdf(t[j][i]);
					}
				}
			}
			
			// �w�K���ʂ̕ۑ�
			save_model();
		}

		std::cout << r << ":\t" << calc_perplexity() << " (" << timer.elapsed() << "[s])" << endl;
	}
}


void TOT::save_psi(const string &file_psi)
{
	ofstream ofs(file_psi);

	for(int k=0; k<K; ++k){
		ofs << k << "\t" << psi[k].alpha << "\t" << psi[k].beta<< endl;
	}
}

void TOT::save_model(void)
{
	const string file_phi = "phi.txt";
	const string file_theta = "theta.txt";
	const string file_psi = "psi.txt";
	
	save_phi(file_phi, 50);
	save_theta(file_theta, 10);
	save_psi(file_psi);
}