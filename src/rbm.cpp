#include "deepnet.hpp"

		
RBM::RBM(int visible, int hidden) {
		size[0] = visible; size[1] = hidden;
		W = W.Random(size[1], size[0])*0.1;
		delta_W.setZero(size[1], size[0]);
		vBias = vBias.Random(size[0])*0.1;
		delta_vBias.setZero(size[0]);
		hBias = hBias.Random(size[1])*0.1;
		delta_hBias.setZero(size[1]);
		}

RBM::RBM(MatrixXd _W, MatrixXd _delta_W, 
	VectorXd _vBias, VectorXd _delta_vBias,
	VectorXd _hBias, VectorXd _delta_hBias, 
	Vector2i _size) {
		size = _size;
		W = _W;
		delta_W = _delta_W;
		vBias = _vBias;
		delta_vBias = _delta_vBias;
		hBias = _hBias;
		delta_hBias = _delta_hBias;
		}

MatrixXd RBM::prob_v_given_h(MatrixXd h) {
	return sigm(h*W + vBias.transpose().replicate(h.rows(), 1));
	}

MatrixXd RBM::prob_h_given_v(MatrixXd v) {
	return sigm(v*W.transpose() + hBias.transpose().replicate(v.rows(), 1));
	}

MatrixXd RBM::sample_v_given_h(MatrixXd h) {
	return bernoulli_sample(prob_v_given_h(h));
	}

MatrixXd RBM::sample_h_given_v(MatrixXd v) {
	return bernoulli_sample(prob_h_given_v(v));
	}

double RBM::train_a_batch(MatrixXd &v1, double learning_rate, double momentum, int CD) {
	int n_sample = v1.rows();
	MatrixXd h1 = bernoulli_sample(prob_h_given_v(v1));

	MatrixXd vn(v1), hn(h1);

	for(int i = 0; i < CD; i++) {
		vn = prob_v_given_h(hn);
		hn = prob_h_given_v(vn);
		if (i < CD - 1) {
			hn = bernoulli_sample(hn);
			}
		}

	MatrixXd delta_W_new = (h1.transpose() * v1 - hn.transpose() * vn) / n_sample;
	delta_W = learning_rate * delta_W_new + momentum * delta_W; 
	W += delta_W;

	VectorXd delta_vBias_new = (v1 - vn).colwise().mean();
	delta_vBias = learning_rate * delta_vBias_new + momentum * delta_vBias;
	vBias += delta_vBias;

	VectorXd delta_hBais_new = (h1 - hn).colwise().mean();
	delta_hBias = learning_rate * delta_hBais_new + momentum * delta_hBias;
	hBias += delta_hBias;

	return (v1 - vn).array().square().sum() / n_sample; // error
	}

// train can be used to train for a certern number of ephoch, and to continue traning
VectorXd RBM::train(MatrixXd &x, int numepochs, int batchsize, double learning_rate, double momentum, double learning_rate_scale, int CD) {
	long n_sample = x.rows();
	if (batchsize > n_sample) batchsize = n_sample;
	int n_batch = n_sample / batchsize; // truncated if not divided
	int remainder = n_sample - n_batch*batchsize;

	int n_batch2 = n_batch; // n_batch2 is the actual batch number
	if (remainder > 0) n_batch2++;

	int s = 0;
	VectorXd error;
	error.setConstant(numepochs*n_batch2, -1);
	PermutationMatrix<Dynamic, Dynamic> perm(n_sample);

	for (int i = 0; i < numepochs; i++) {
		perm.setIdentity();
		random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		MatrixXd x_perm = perm * x;
		int j = 0;
		if (n_batch >= 1) {
			for (; j < n_sample - remainder; j += batchsize) {
				error[s] = train_a_batch(x_perm.block(j, 0, batchsize, x.cols()), learning_rate, momentum, CD);
				s++;
				}
			}
		if (remainder > 0) {
			error[s] = train_a_batch(x_perm.block(j, 0, remainder, x.cols()), learning_rate, momentum, CD);
			s++;
			}
		learning_rate = learning_rate * learning_rate_scale;
		}
	return error;
	}


// The following functions are for Rcpp
/*
List C_RBM(MatrixXd x, int hidden, int numepochs = 3, int batchsize = 100, 
	double learning_rate = 0.8, double learning_rate_scale = 1, int cd = 1,
	double momentum = 0.5, string visible_type = "bin", string hidden_type = "bin"
	) {
	RBM rbm(xx.cols(), hidden, learning_rate, learning_rate_scale, 
		momentum, visible_type, hidden_type, cd);
	VectorXd error = rbm.train(x);
	return List::create(
		Named("W") = rbm.W,
		Named("vBias") = rbm.vBias,
		Named("hBias") = rbm.hBias,
		Named("delta_W") = rbm.delta_W,
		Named("delta_vBias") = rbm.delta_vBias,
		Named("delta_hBias") = rbm.delta_hBias,
		Named("size") = rbm.size,
		Named("learning_rate") = rbm.learning_rate,
		Named("learning_rate_scale") = rbm.learning_rate_scale,
		Named("momentum") = rbm.momentum,
		Named("hidden_type") = rbm.hidden_type, 
		Named("visible_type") = rbm.visible_type,
		Named("CD") = rbm.CD,
		Named("error") = error
		);
	}

List C_RBM_MORE(MatrixXd x, MatrixXd W, MatrixXd delta_W, 
		VectorXd vBias, VectorXd delta_vBias,
		VectorXd hBias, VectorXd delta_hBias, 
		Vector2i size, double learning_rate, 
		double learning_rate_scale, double momentum,
		string hidden_type, string visible_type, int cd
	) {
	RBM rbm(W, delta_W, vBias, delta_vBias, hBias, delta_hBias, 
		size, learning_rate, learning_rate_scale, momentum, 
		hidden_type, visible_type, cd);
	VectorXd error = rbm.train(x);
	return List::create(
		Named("W") = rbm.W,
		Named("vBias") = rbm.vBias,
		Named("hBias") = rbm.hBias,
		Named("delta_W") = rbm.delta_W,
		Named("delta_vBias") = rbm.delta_vBias,
		Named("delta_hBias") = rbm.delta_hBias,
		Named("size") = rbm.size,
		Named("learning_rate") = rbm.learning_rate,
		Named("learning_rate_scale") = rbm.learning_rate_scale,
		Named("momentum") = rbm.momentum,
		Named("hidden_type") = rbm.hidden_type, 
		Named("visible_type") = rbm.visible_type,
		Named("CD") = rbm.CD,
		Named("error") = error
		);
	}

MatrixXd C_RBM_PREDICT(MatrixXd x, string neuron = 'h',  type = 'prob',
		MatrixXd W, MatrixXd delta_W, 
		VectorXd vBias, VectorXd delta_vBias,
		VectorXd hBias, VectorXd delta_hBias, 
		Vector2i size, double learning_rate, 
		double learning_rate_scale, double momentum,
		string hidden_type, string visible_type, int cd
	) {
	RBM rbm(W, delta_W, vBias, delta_vBias, hBias, delta_hBias, 
		size, learning_rate, learning_rate_scale, momentum, 
		hidden_type, visible_type, cd);

	MatrixXd out;
	if (neuron == 'h') {
		out = rbm.prob_h_given_v(x);
		if (type == 'sample') {
			out = rbm.bernoulli_sample(out);
			}
	} else {
		out = rbm.prob_v_given_h(x);
		}
	return out;
	}
*/
