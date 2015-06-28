#include "deepnet.hpp"

int feedForwardNetwork::bp(MatrixXd error, vector<MatrixXd> & dW, vector<VectorXd> & dB, double learning_rate, double momentum) {
	int layer = layer_size.size() - 1;
	MatrixXd d(error);
	//d: propogating error

	errorBackPropFromOutput(d);

	for (int i = layer-1; i >= 0; i--) {
 		dW[i] = dW[i]*momentum + (d * post[i].transpose()) * (learning_rate / d.cols()); // momentum + dW -> new dW
		dB[i] = dB[i]*momentum + d.rowwise().mean() * learning_rate; // momentum + dB -> new dB

		if (i > 0) errorBackProp(d, i);

		W[i].noalias() -= dW[i];
		B[i].noalias() -= dB[i];
		}
	return 0;
	}


VectorXd feedForwardNetwork::bpTrain(MatrixXd x, MatrixXd y, int numepochs, int batchsize, double learning_rate, double momentum, double learning_rate_scale, bool verbose) {
	// initialize training parameters
	std::vector<MatrixXd> dW(layer_size.size()-1);
	std::vector<VectorXd> dB(layer_size.size()-1);
	for(int i = 0; i < layer_size.size()-1; i++) {
		dW[i].setZero(layer_size[i+1], layer_size[i]);
		dB[i].setZero(layer_size[i+1]);
		}

	long n_sample = x.cols();
	if (batchsize > n_sample) batchsize = n_sample;
	int n_batch = n_sample / batchsize; // truncated if not divided
	int remainder = n_sample - n_batch*batchsize;

	int n_batch2 = n_batch; // n_batch2 is the actual batch number
	if (remainder > 0) n_batch2++;

	int s = 0;  // update iteration, total iteration = numepoch x numbatch
	VectorXd loss(numepochs*n_batch2);  // mean sum of square error/loss
	MatrixXd error;  //raw error: per sample per output dimension
	error.setConstant(numepochs, n_batch2, -1);
	PermutationMatrix<Dynamic, Dynamic> perm(n_sample);

	MatrixXd x_perm(x);
	MatrixXd y_perm(y);

	for (int i = 0; i < numepochs; i++) {
		if (verbose) cout <<  "Epoch " << i + 1 << endl;
		perm.setIdentity();
		random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		x_perm = x_perm * perm;  // col = sample, shuffle samples
		y_perm = y_perm * perm;
		int this_batchsize = batchsize;

		for(int j = 0; j < n_sample; j +=batchsize) {
			if (j >= n_sample - remainder) this_batchsize = remainder;
			error = ff(x_perm.middleCols(j, this_batchsize), y_perm.middleCols(j,  this_batchsize));
			bp(error, dW, dB, learning_rate, momentum);
			if (output == "softmax") {
				loss[s] = -(y_perm.middleCols(j, this_batchsize).array() * post[layer_size.size()-1].array().log()).colwise().sum().mean();
			} else {
				loss[s] = error.array().square().mean();
				}
			s++;
			}
		learning_rate *= learning_rate_scale;
		}
	return loss;
	}
