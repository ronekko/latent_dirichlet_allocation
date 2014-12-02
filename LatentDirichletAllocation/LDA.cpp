#include "stdafx.h"
#include "LDA.h"
#include "utility.hpp"

using namespace std;


LDA::~LDA(void)
{
}


LDA::LDA(const int &K_, const double &alpha_total_mass_, const double &beta_total_mass_, int n_iter_, enum Method method_, const int &seed)
	: K(K_), alpha_total_mass(alpha_total_mass_), beta_total_mass(beta_total_mass_), n_iter(n_iter_), method(method_)
{
	rng.seed(seed != -1 ? seed : std::random_device()());
}


void LDA::fit(vector<std::unordered_map<int, int>> bows)
{
	M = bows.size();
	// W is the vocabulary's size since accumulate function calculates the biggest key in keys in all unordered_maps in the vector
	W = 1 + boost::accumulate(bows, 0, [](int current_max, const unordered_map<int, int> &bow){
		return max(current_max, (*boost::max_element(bow)).first);
	});

	// The "bow" object (counts of wordtypes) is converted into a sequence of word tokens (x_j object).
	// For example, bow:{(0, 2), (1, 3), (3, 2)} -> x_j:{0, 0, 1, 1, 1, 3, 3}
	for (const auto &bow : bows){
		int N_j = boost::accumulate(bow, 0, [](int sum, const pair<int, int> &type_to_count){
			return sum + type_to_count.first;
		});
		vector<int> x_j;
		for (const auto &type_count : bow){
			int word_type = type_count.first;
			int word_count = type_count.second;
			for (int c = 0; c < word_count; ++c){
				x_j.push_back(word_type);
			}
		}
		boost::random_shuffle(x_j); // shuffle the ordering of tokens in j-th document
		x.push_back(x_j);
	}

	// initialize z and associated counts
	z = vector<vector<int>>(M);
	n_k = vector<int>(K, 0);
	n_wk = vector<vector<int>>(W, vector<int>(K, 0));
	n_jk = vector<vector<int>>(M, vector<int>(K, 0));
	boost::uniform_int<> uint(0, K-1);
	for(int j=0; j<M; ++j){
		int N_j = x[j].size();
		z[j] = vector<int>(N_j);
		for(int i=0; i<N_j; ++i){
			int k = uint(rng);
			z[j][i] = k;

			int w = x[j][i];
			n_k[k]++;
			n_wk[w][k]++;
			n_jk[j][k]++;
		}
	}

	// set hyperparameters
	alpha_k = alpha_total_mass / K;
	beta_w = beta_total_mass / W;
	N = 0;
	for(auto &x_j : x){
		N += x_j.size();
	}

	//
	cout << "M = " << M << " (number of documents)" << endl
		 << "N = " << N << " (number of tokens in the corpus)" << endl
		 << "W = " << W << " (size of the vocabulary)" << endl
		 << "K = " << K << " (number of topics)" << endl
		 << "alpha_k = " << alpha_k << endl
		 << "beta_w = " << beta_w << endl;

	switch (method){
	case Method::CGS: train_by_CGS(n_iter); break;
	}
}


void LDA::train_by_CGS(const int &n_iter)
{
	for(int r=0; r<n_iter; ++r)
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
					p[k] = (n_jk[j][k] + alpha_k) * (n_wk[w][k] + beta_w) / (n_k[k] + W * beta_w);
				}
				int k_new = util::multinomialByUnnormalizedParameters(rng, p);
				
				z[j][i] = k_new;

				n_k[k_new]++;
				n_wk[w][k_new]++;
				n_jk[j][k_new]++;
			}
		}
		cout << r << ":\t" << calc_perplexity() << "(" << timer.elapsed() << ")" << endl;
	}
}



// phi[K][W]
vector<vector<double>> LDA::calc_phi(void)
{
	vector<vector<double>> phi(K, vector<double>(W));
	for(int k=0; k<K; ++k){
		double denomi = n_k[k] + W * beta_w;
		for(int w=0; w<W; ++w){
			phi[k][w] = (n_wk[w][k] + beta_w) / denomi;
		}
	}

	return phi;
}


// theta[M][K]
vector<vector<double>> LDA::calc_theta(void)
{
	vector<vector<double>> theta(M, vector<double>(K));
	for(int j=0; j<M; ++j){
		double denomi = boost::accumulate(n_jk[j], 0.0) + K * alpha_k;
		for(int k=0; k<K; ++k){
			theta[j][k] = (n_jk[j][k] + alpha_k) / denomi;
		}
	}

	return theta;
}


double LDA::calc_perplexity(void)
{
	vector<vector<double>> phi = calc_phi();
	vector<vector<double>> theta = calc_theta();
	double log_sum = 0.0;

	for(int j=0; j<M; ++j)
	{
		int N_j = x[j].size();
		for(int i=0; i<N_j; ++i)
		{
			int w = x[j][i];
			double px = 0.0;
			for(int k=0; k<K; ++k){
				px += theta[j][k] * phi[k][w];
			}
			log_sum += log(px);
		}
	}

	double perplexity = exp(-log_sum / static_cast<double>(N));
	return perplexity;
}


void LDA::save_phi(const string &file_phi, int W_top)
{
	if(W_top == 0){ W_top = W; }
	ofstream ofs(file_phi);
	auto phi = calc_phi();
	for(int k=0; k<K; ++k){
		ofs << k << endl;
		vector<pair<double, int>> phi_k(W);
		for(int w=0; w<W; ++w){
			phi_k[w] = make_pair(phi[k][w], w);
		}

		boost::sort(phi_k, greater<pair<double, int>>());

		for(int w=0; w<W_top; ++w){
			ofs << phi_k[w].first << "\t" << vocabulary[phi_k[w].second] << endl;
		}
	}
}

void LDA::save_theta(const string &file_theta, int K_top)
{
	if(K_top == 0){ K_top = K; }
	ofstream ofs(file_theta);
	auto theta = calc_theta();
	for(int j=0; j<M; ++j){
		ofs << j << endl;
		vector<pair<double, int>> theta_j(K);
		for(int k=0; k<K; ++k){
			theta_j[k] = make_pair(theta[j][k], k);
		}

		boost::sort(theta_j, greater<pair<double, int>>());
		
		for(int k=0; k<K_top; ++k){
			ofs << theta_j[k].first << "\t" << theta_j[k].second << endl;
		}
	}
}

void LDA::save_model(const std::string &file_phi, const std::string &file_theta, const int &W_top, const int &K_top)
{	
	save_phi(file_phi, W_top);
	save_theta(file_theta, K_top);
}


std::vector<std::unordered_map<int, int>> LDA::load_bow_file(const std::string &file_bow)
{
	// load corpus file which is a collection of bag-of-words' for each document
	std::vector<std::unordered_map<int, int>> bows;
	ifstream ifs_bow(file_bow);
	string line;
	while (getline(ifs_bow, line))
	{
		vector<string> tokens;
		boost::algorithm::split(tokens, line, boost::algorithm::is_space());

		std::unordered_map<int, int> word_count_map;
		for (string &token : tokens){
			int word, count;
			sscanf_s(token.c_str(), "%d:%d", &word, &count);
			word_count_map[word] = count;
		}
		bows.push_back(word_count_map);
	}
	return bows;
}
