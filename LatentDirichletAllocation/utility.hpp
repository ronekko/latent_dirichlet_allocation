#pragma once

#include "stdafx.h"

namespace util
{

using namespace std;

// 画像サイズをscale倍に拡大する。元画像の1ピクセルが、拡大後の画像で一辺scale [pixel]の正方形となるように拡大する。つまり平滑化しない。
inline cv::Mat upsample(const cv::Mat &image, const int &scale)
{
	int w = scale * image.cols;
	int h = scale * image.rows;
	int depth = image.depth();
	int channels = image.channels();
	cv::Mat result(h, w, CV_MAKETYPE(depth, channels));

	for(int y=0; y<h; ++y){
		for(int x=0; x<w; ++x){
			for(int c=0; c<channels; ++c){
				int xx = x / scale;
				int yy = y / scale;
				int ww = w / scale;
				result.data[(y * w + x) * channels + c] = image.data[(yy * ww + xx) * channels + c];
			}
		}
	}
	return result;
}



// shapeかscaleがほぼ0のときは常に0を返す（ガンマ分布の定義は 0<shape, 0<scale であるため）
inline double gammaRandom(boost::mt19937 &engine, const double &shape, const double &scale)
{
	if(shape < 1.0e-300 || scale < 1.0e-300){ return 0; }
	boost::math::gamma_distribution<> dist(shape, scale);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}


// ディリクレ分布から乱数を生成する http://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution
inline std::vector<double> dirichletRandom(boost::mt19937 &engine, const std::vector<double> &alpha)
{	
	const int K = alpha.size();
	std::vector<double> y(K);
	double sumY = 0.0;

	for(int k=0; k<K; ++k){
		//y[k] = boost::gamma_distribution<>(alpha[k], 1.0)(engine);	// shapeパラメータが大きいと落ちる
		y[k] = gammaRandom(engine, alpha[k], 1.0);
		sumY += y[k];
	}

	for(int k=0; k<K; ++k){
		y[k] /= sumY;
	}

	return y;
}


inline double betaRandom(boost::mt19937 &engine, const double &alpha, const double &beta)
{
	boost::math::beta_distribution<> dist(alpha, beta);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}

// 多項分布からのサンプリング、ただしパラメータは正規化されていない（\sum p_iが1とは限らない）
inline int multinomialByUnnormalizedParameters(boost::mt19937 &engine, const vector<double> &p)
{
	const int K=p.size();
	vector<double> CDF(K);
	double z = 0.0;
	for(int k=0; k<K; ++k){
		CDF[k] = z + p[k];
		z = CDF[k];
	}

	double u = boost::uniform_01<>()(engine) * CDF.back();
	for(int k=0; k<K; ++k){
		if(u < CDF[k]){
			return k;
		}
	}
	cout <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	return K-1;
}

inline double logsumexp (double x, double y, bool flg)
{
	if (flg) return y; // init mode
	if (x == y) return x + 0.69314718055; // log(2)
	double vmin = std::min (x, y);
	double vmax = std::max (x, y);
	if (vmax > vmin + 50) {
		return vmax;
	} else {
		return vmax + std::log (std::exp (vmin - vmax) + 1.0);
	}
}
// 多項分布からのサンプリング、ただしパラメータはlog(p_1), ... , log(p_K)で与えられ、正規化されていない（\sum p_iが1とは限らない）
inline int multinomialByUnnormalizedLogParameters(boost::mt19937 &engine, const vector<double> &lnp)
{
	const int K=lnp.size();
	vector<double> logCDF(K);
	double z = 0.0;
	for(int k=0; k<K; ++k){
		z = logsumexp(z, lnp[k], (k==0));
		logCDF[k] = z;
	}

	double u = log(boost::uniform_01<>()(engine)) + logCDF.back();
	for(int k=0; k<K; ++k){
		if(u < logCDF[k]){
			return k;
		}
	}

	return K-1;
}

// Chinese restaurant table distributionからの乱数生成
inline int CRTRandom(boost::mt19937 &engine, const int &n, const double &alpha)
{
	if(n < 1){ return 0; }
	if(alpha == 0){ return 1;}

	int l=1;
	for(int i=1; i<n; ++i){
		double p = alpha / (i + alpha);
		l += boost::bernoulli_distribution<>(p)(engine);
	}
	return l;
}

inline void test_CRTRandom(void)
{
	boost::mt19937 engine(0);
	double alpha = 4.0;
	int r = 100000000;
	{
		//int n = 5; double pmf[] = {0, 24, 50, 35, 10, 1};
		//int n = 6; double pmf[] = {0, 120, 274, 225, 85, 15, 1};
		//int n = 7; double pmf[] = {0, 720, 1764, 1624, 735, 175, 21, 1};
		//int n = 8; double pmf[] = {0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1};
		int n = 9; double pmf[] = {0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1};
		for(int l=1; l<=n; ++l){	pmf[l] *= pow(alpha, l); }
		double total = boost::accumulate(pmf, 0.0);
		for(int l=1; l<=n; ++l){	pmf[l] /= total; }

		vector<int> result(n+1, 0);
		for(int i=0; i<r; ++i){
			int l = CRTRandom(engine, n, alpha);
			result[l]++;
		}

		vector<double> prop(n+1, 0.0);
		for(int l=1; l<=n; ++l){
			prop[l] = result[l] / static_cast<double>(r);
		}
		
		cout << "empirical\tpmf[l]\t\tdiff" << endl;
		for(int l=0; l<=n; ++l){
			cout << prop[l] << "    \t" << pmf[l] << "    \t" << prop[l] - pmf[l] << endl;
		}
	}
}

class BetaDistribution{
public:
	BetaDistribution(const double &alpha, const double &beta) : alpha(alpha), beta(beta){
		using boost::math::lgamma;
		log_normalization_term = lgamma(alpha + beta) - (lgamma(alpha) + lgamma(beta));
	};
	~BetaDistribution(void){};
	inline double log_pdf(const double &x){
		return log_normalization_term + (alpha - 1.0) * log(x) + (beta - 1.0) * log(1.0 - x);
	};
private:
	const double alpha;
	const double beta;
	double log_normalization_term;
};
};