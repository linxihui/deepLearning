#include "deepnet.hpp"

int feedForwardNetwork::rprop(MatrixXd error, vector<MatrixXd> & dW, vector<VectorXd> & dB, 
	vector<ArrayXXi> & signDeltaW, vector<ArrayXi> & signDeltaB, 
	double incScale, double decScale, double incScaleMax, double decScaleMin) {

	int layer = layer_size.size() - 1;
	MatrixXd d(error); // propogating error
	ArrayXXi signDeltaW_new;
	ArrayXi signDeltaB_new;
	ArrayXXi OldxNewW;
	ArrayXi OldxNewB;

	errorBackPropFromOutput(d);

	for (int i = layer-1; i >= 0; i--) {

 		signDeltaW_new = sign((d * post[i].transpose()).array()); 
		signDeltaB_new = sign((d.rowwise().mean()).array());
		OldxNewW = signDeltaW[i]*signDeltaW_new;
		OldxNewB = signDeltaB[i]*signDeltaB_new;

		dW[i] = (OldxNewW > 0).select((dW[i]*incScale).eval().cwiseMin(incScaleMax), 0.) + 
			(OldxNewW < 0).select((dW[i]*decScale).eval().cwiseMax(decScaleMin), 0.) + 
			(OldxNewW == 0).select(dW[i], 0.);
		dB[i] = (OldxNewB > 0).select((dB[i]*incScale).eval().cwiseMin(incScaleMax), 0.) + 
			(OldxNewB < 0).select((dB[i]*decScale).eval().cwiseMax(decScaleMin), 0.) +
			(OldxNewB == 0).select(dB[i], 0.);

		// prop-
		signDeltaW[i] = signDeltaW_new; 
		signDeltaB[i] = signDeltaB_new; 

		// iprop-
		// signDeltaW[i] = (OldxNewW >= 0).select(signDeltaW_new, 0);
		// signDeltaB[i] = (OldxNewB >= 0).select(signDeltaB_new, 0);
		 
		if (i > 0) errorBackProp(d, i);

		W[i].noalias() -= dW[i].cwiseProduct(signDeltaW[i].matrix().cast<double>());
		B[i].noalias() -= dB[i].cwiseProduct(signDeltaB[i].matrix().cast<double>());
		} 
	return 0; 
	}


VectorXd feedForwardNetwork::rpropTrain(MatrixXd x, MatrixXd y, int numepochs, int batchsize, double incScale, double decScale, double incScaleMax, double decScaleMin, bool verbose) {
	// initialize training parameters
	std::vector<MatrixXd> dW(layer_size.size()-1);
	std::vector<VectorXd> dB(layer_size.size()-1);
	std::vector<ArrayXXi> signDeltaW(layer_size.size()-1);
	std::vector<ArrayXi> signDeltaB(layer_size.size()-1);

	for(int i = 0; i < layer_size.size()-1; i++) {
		dW[i].setConstant(layer_size[i+1], layer_size[i], 0.1);
		dB[i].setConstant(layer_size[i+1], 0.1);
		signDeltaW[i].setZero(layer_size[i+1], layer_size[i]);
		signDeltaB[i].setZero(layer_size[i+1]);
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
			rprop(error, dW, dB, signDeltaW, signDeltaB, incScale, decScale, incScaleMax, decScaleMin);
			if (output == "softmax") {
				loss[s] = -(y_perm.middleCols(j, this_batchsize).array() * post[layer_size.size()-1].array().log()).colwise().sum().mean();
			} else {
				loss[s] = error.array().square().mean();
				}
			s++;
			}
		}
	return loss;
	}
