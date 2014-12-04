// LatentDirichletAllocation : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "LDA.h"
#include "TOT.h"

using namespace std;



void run_demo(const int &size=5)
{
	int V = size;
	int W = V * V;
	int K = V * 2 + 2;
	double alpha_total_mass = 1.0;
	double beta_total_mass = 10.0;
	int seed = -1;
	int M = 500;
	double mean_N_j = 200.0;

	///////////////////////////////////////////////////////////////////
	// generate a corpus (forward pass of the generative model)
	vector<double> alpha(K, alpha_total_mass / K);
	vector<double> beta(K, beta_total_mass / W);
	boost::mt19937 rng;
	std::poisson_distribution<> poisson(mean_N_j);

	// make artificial topics, instead of generating phi_k ~ Dir(beta_1, ... , beta_W)
	vector<vector<double>> phi(K, vector<double>(W));
	for (int v = 0; v < V; ++v){
		for (int k = 0; k < (K - 2) / 2; ++k) { phi[k][k * V + v] = 0.2; }
		for (int k = (K - 2) / 2; k < K - 2; ++k) { phi[k][v * V + k - V] = 0.2; }
		phi[K - 2][v * (V+1)] = 0.2;
		phi[K - 1][(v + 1) * (V-1)] = 0.2;
	}
	util::show_topics("true phi", phi);
	vector<boost::random::discrete_distribution<>> topics(phi.begin(), phi.end());

	// generate documents
	vector<vector<int>> corpus;
	for (int j = 0; j < M; ++j)
	{
		int N_j = poisson(rng);
		vector<int> doc; // j-th document
		vector<double> theta_j = util::dirichletRandom(rng, alpha); // generate theta_j ~ Dir(alpha_1, ... , alpha_K)
		boost::random::discrete_distribution<> multi(theta_j);
		//generate i-th token in the j-th document
		for (int i = 0; i < N_j; ++i){
			int k = multi(rng);     // generate z_ji ~ Multi(theta_j)
			int w = topics[k](rng); // generate x_ji ~ Multi(phi_{z_i})
			doc.push_back(w);
		}
		corpus.push_back(doc);
	}
	///////////////////////////////////////////////////////////////////

	// make BoW
	vector<std::unordered_map<int, int>> x;
	for (auto &doc : corpus){
		std::unordered_map<int, int> x_j;
		for (auto &token : doc){
			x_j[token]++;
		}
		x.push_back(x_j);
	}

	// learning by CGS
	LDA lda(K, alpha_total_mass, beta_total_mass, 1, LDA::Method::CGS, seed);
	lda.fit(x);
	for (int iter = 1; iter < 3000; ++iter){
		cout << iter << ": ";
		lda.train_by_CGS(1);
		auto phi = lda.calc_phi();
		util::show_topics("estimated phi", phi);
	}

	cv::waitKey(0);	
}



int _tmain(int argc, _TCHAR* argv[])
{
#define DEMO // comment out this line to run another demo which uses a real corpus

#ifdef DEMO
	run_demo(5); // set size of the demo

#else
	const string file_bow = "NIPS0-12/counts0-12.txt";
	const string file_vocabulary = "NIPS0-12/voca0-12.txt";
	const string file_timestamp = "NIPS0-12/pclass0-12.txt";

	const int K = 60;
	const double alpha = 1.0;
	const double beta = 20.0;
	const int num_iter = 300;
	const auto method = LDA::Method::CVB0;
	const int seed = 0;

	LDA lda(K, alpha, beta, num_iter, method, seed);
	//TOT lda(file_bow, file_vocabulary, file_timestamp, 60, 0);

	vector<std::unordered_map<int, int>> bows = LDA::load_bow_file(file_bow);
	lda.fit(bows);
	
	lda.save_model("result_phi.txt", "result_theta.txt", 50, 10, file_vocabulary);

	lda.n_iter = 300;
	auto theta_test = lda.transform(bows);
	lda.save_model("result_test_phi.txt", "result_test_theta.txt", 20, 10, file_vocabulary);

#endif
	return 0;
}

