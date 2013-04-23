#include "stdafx.h"
#include "TOT.h"
#include "utility.hpp"
#include <omp.h>

using namespace std;

TOT::TOT(void) : LDA()
{
}


TOT::~TOT(void)
{
}

TOT::TOT(const string &file_bow, const string &file_vocabulary, const string &file_timestamp)
		: LDA(file_bow, file_vocabulary)
{
	// コーパスのタイムスタンプファイルのロード
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
	// たとえば、タイムスタンプが2000～2010なら11年間あるのでmax_timestamp+1する
	double range_timestamp = max_timestamp - min_timestamp + 1.0;

	t = vector<vector<double>>(M);
	for(int j=0; j<M; ++j){
		int N_j = x[j].size();
		t[j] = vector<double>(N_j);
		for(int i=0; i<N_j; ++i){
			double timestamp = timestamps[j];
			// 整数のタイムスタンプに対応する区間の中央の値（一年間の中央）を文書のタイムスタンプとする
			double normalized_timestamp = (timestamp - min_timestamp + 0.5) / range_timestamp;
			t[j][i] = normalized_timestamp;
		}
	}

	// パラメタ、尤度のキャッシュの初期化
	psi = vector<pair<double, double>>(K, pair<double, double>(1.0, 1.0));
	
	beta_likelihood = vector<vector<double>>(M);
	vector<boost::math::beta_distribution<>> beta_distributions(K);
	for(int k=0; k<K; ++k){
		beta_distributions[k] = boost::math::beta_distribution<>(psi[k].first, psi[k].second);
	}
#pragma omp parallel for
	for(int j=0; j<M; ++j){
		int N_j = t[j].size();
		beta_likelihood[j] = vector<double>(N_j);
		for(int i=0; i<N_j; ++i){
			for(int k=0; k<K; ++k){
				beta_likelihood[j][i] = boost::math::pdf(beta_distributions[k], t[j][i]);
			}
		}
	}
}


void TOT::train(const int &iter)
{
	int interval_beta_update = 10;
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
					p[k] = (n_jk[j][k] + ALPHA) * (n_wk[w][k] + BETA) / (n_k[k] + W * BETA) * beta_likelihood[j][i];
				}
				int k_new = util::multinomialByUnnormalizedParameters(rgen, p);
				
				z[j][i] = k_new;

				n_k[k_new]++;
				n_wk[w][k_new]++;
				n_jk[j][k_new]++;
			}
		}
		
		// ベータ分布パラメタと尤度の更新（iterがinterval_beta_update回ごとに）
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
				t_mean[k] /= static_cast<double>(n_k[k]);
			}
			for(int j=0; j<M; ++j){
				for(int i=0; i<t[j].size(); ++i){
					int k = z[j][i];
					t_var[k] += (t[j][i] - t_mean[k]) * (t[j][i] - t_mean[k]);
				}
			}
			for(int k=0; k<K; ++k){
				// 分散が0になるケースがありえる
				t_var[k] /= static_cast<double>(n_k[k]);
				// TOT論文のAppendixの最後、ψを求める式ふたつの第2項（括弧の中）
				double last_term = t_mean[k] * (1.0 - t_mean[k]) / t_var[k] - 1.0;
				psi[k].first = t_mean[k] * last_term;
				psi[k].second = (1.0 - t_mean[k]) * last_term;
			}
		
			// ベータ分布尤度項の更新
			vector<boost::math::beta_distribution<>> beta_distributions(K);
			for(int k=0; k<K; ++k){
				beta_distributions[k] = boost::math::beta_distribution<>(psi[k].first, psi[k].second);
			}
#pragma omp parallel for
			for(int j=0; j<M; ++j){
				for(int i=0; i<t[j].size(); ++i){
					for(int k=0; k<K; ++k){
						beta_likelihood[j][i] = boost::math::pdf(beta_distributions[k], t[j][i]);
					}
				}
			}
		}

		std::cout << r << ":\t" << calc_perplexity() << " (" << timer.elapsed() << "[s])" << endl;
	}
}