#include "deepnet.hpp"

DBN::DBN(int in_dim, VectorXi hid) {
	input_dim = in_dim;
	hidden = hid;
	VectorXi vis_hid(hidden.size() + 1);
	vis_hid[0] = input_dim;
	vis_hid.segment(1, hidden.size()) = hidden;
	layer.reserve(hidden.size());
	for (int i = 0; i < hidden.size(); i++) {
		layer.push_back(RBM(vis_hid[i], vis_hid[i+1]));
		}
	}


MatrixXd DBN::train(MatrixXd &x, double learning_rate, double learning_rate_scale, 
	double momentum, int numepochs, int batchsize, int CD, bool verbose) {
	MatrixXd pre_layer(x);
	for (int i = 0; i < hidden.size(); i++) {
		if (verbose) cout << "Training layer " << i+1 << "..." << endl;
		layer[i].train(pre_layer, numepochs, batchsize, learning_rate, momentum, learning_rate_scale, CD);
		pre_layer = layer[i].prob_h_given_v(pre_layer);
		}
	return pre_layer;
	}


MatrixXd DBN::extractHiddenFeature(MatrixXd v) {
	MatrixXd pre_layer(v);
	for (int i = 0; i < hidden.size(); i++) {
		cout << "i = " << i << endl;
		pre_layer = layer[i].prob_h_given_v(pre_layer);
		}
	return pre_layer;
	}

MatrixXd DBN::generateFromHidden(MatrixXd h) {
	MatrixXd pre_layer(h);
	for (int i = hidden.size()-1; i >= 0; i--) {
		cout << "i = " << i << endl;
		pre_layer = layer[i].prob_v_given_h(pre_layer);
		}
	cout << "Done" << endl;
	return pre_layer;
	}

MatrixXd DBN::reconstructFromInput(MatrixXd x) {
	MatrixXd pre_layer = extractHiddenFeature(x);
	return generateFromHidden(pre_layer);
	}
