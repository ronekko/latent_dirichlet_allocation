#include "stdafx.h"
#include "LDA.h"
#include "utility.hpp"

using namespace std;

LDA::LDA(void)
{
}


LDA::~LDA(void)
{
}


LDA::LDA(const string &file_bow, const string &file_vocabulary)
{
	rgen.seed(0);
	K = 70;

	// コーパスのカウントファイルのロード
	ifstream ifs_bow(file_bow);
	string line;
	M = 0;
	while(getline(ifs_bow, line))
	{
		using namespace boost::algorithm;
		vector<string> tokens;
		
		split(tokens, line, is_space());
		
		int L = tokens.size(); // 文書中のユニーク単語数
		vector<int> words(L);
		vector<int> counts(L, 0);

		for(int l=0; l<L; ++l){
			sscanf_s(tokens[l].c_str(), "%d:%d", &(words[l]), &(counts[l]));
		}

		int N_j = boost::accumulate(counts, 0);
		vector<int> x_j(N_j);
		for(int l=0, i=0; l<L; ++l){
			int word = words[l];
			for(int c=0; c<counts[l]; ++c){
				x_j[i] = word;
				++i;
			}
		}
		boost::random_shuffle(x_j); // 文書内のトークンの出現順をシャッフルする
		x.push_back(x_j);
		++M;
	}
	
	// コーパスの語彙のロード
	ifstream ifs_vocabulary(file_vocabulary);
	W = 0;
	while(getline(ifs_vocabulary, line))
	{
		boost::algorithm::trim(line);
		vocabulary.push_back(line);
		++W;
	}
	
	// zの初期化、カウントの初期化
	z = vector<vector<int>>(M);
	n_k = vector<int>(K, 0);
	n_wk = vector<vector<int>>(W, vector<int>(K, 0));
	n_jk = vector<vector<int>>(M, vector<int>(K, 0));
	boost::uniform_int<> uint(0, K-1);
	for(int j=0; j<M; ++j){
		int N_j = x[j].size();
		z[j] = vector<int>(N_j);
		for(int i=0; i<N_j; ++i){
			int k = uint(rgen);
			z[j][i] = k;

			int w = x[j][i];
			n_k[k]++;
			n_wk[w][k]++;
			n_jk[j][k]++;
		}
	}

	// パラメタなどの初期化
	ALPHA = 1.0 / K;
	BETA = 0.001;
	N = 0;
	for(auto &x_j : x){
		N += x_j.size();
	}

	//
	cout << "M = " << M << " (number of documents)" << endl
		 << "N = " << N << " (number of tokens in the corpus)" << endl
		 << "W = " << W << " (size of the vocabulary)" << endl
		 << "K = " << K << " (number of topics)" << endl
		 << "α= " << ALPHA << endl
		 << "β= " << BETA << endl;
}



void LDA::train(const int &iter)
{
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
					p[k] = (n_jk[j][k] + ALPHA) * (n_wk[w][k] + BETA) / (n_k[k] + W * BETA);
				}
				int k_new = util::multinomialByUnnormalizedParameters(rgen, p);
				
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
		double denomi = n_k[k] + W * BETA;
		for(int w=0; w<W; ++w){
			phi[k][w] = (n_wk[w][k] + BETA) / denomi;
		}
	}

	return phi;
}


// theta[M][K]
vector<vector<double>> LDA::calc_theta(void)
{
	vector<vector<double>> theta(M, vector<double>(K));
	for(int j=0; j<M; ++j){
		double denomi = boost::accumulate(n_jk[j], 0.0) + K * ALPHA;
		for(int k=0; k<K; ++k){
			theta[j][k] = (n_jk[j][k] + ALPHA) / denomi;
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

void LDA::save_model(void)
{
	const string file_phi = "phi.txt";
	const string file_theta = "theta.txt";
	
	save_phi(file_phi, 50);
	save_theta(file_theta, 10);
}