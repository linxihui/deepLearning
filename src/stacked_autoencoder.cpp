#include "deepnet.hpp"

stackedAutoEncoder::stackedAutoEncoder(int in_dim, VectorXi hid, string act_fun, string reconst_fun, double hid_dropout, double vis_dropout) {
	input_dim = in_dim;
	hidden = hid;
	VectorXi vis_hid(hidden.size() + 1);
	vis_hid[0] = input_dim;
	vis_hid.segment(1, hidden.size()) = hidden;
	layer.reserve(hidden.size());
	for (int i = 0; i < hidden.size(); i++) {
		VectorXi layer_size(3);
		layer_size << vis_hid[i], vis_hid[i+1], vis_hid[i];
		if (i == 0) {
			layer.push_back(feedForwardNetwork(layer_size, act_fun, reconst_fun, hid_dropout, vis_dropout));
		} else {
			layer.push_back(feedForwardNetwork(layer_size, act_fun, act_fun, hid_dropout, vis_dropout));
			}
		}
	}

MatrixXd stackedAutoEncoder::bpTrain(MatrixXd x, int numepochs, int batchsize, double learning_rate, double momentum, double learning_rate_scale, bool verbose) {
	MatrixXd prelayer_x(x);
	for (int i = 0; i < hidden.size(); i++) {
		if (verbose) cout << "Training layer " << i+1 << "..." << endl;
		layer[i].bpTrain(prelayer_x, prelayer_x, numepochs, batchsize, learning_rate, momentum, learning_rate_scale, false);
		prelayer_x = layer[i].predict(prelayer_x, 1);
		}
	return prelayer_x;
	}

MatrixXd stackedAutoEncoder::rpropTrain(MatrixXd x, int numepochs, int batchsize, double incScale, double decScale, double incScaleMax, double decScaleMin, bool verbose) {
	MatrixXd prelayer_x(x);
	for (int i = 0; i < hidden.size(); i++) {
		if (verbose) cout << "Training layer " << i+1 << "..." << endl;
		layer[i].rpropTrain(prelayer_x, prelayer_x, numepochs, batchsize, incScale, decScale, incScaleMax, decScaleMin, false);
		prelayer_x = layer[i].predict(prelayer_x, 1);
		}
	return prelayer_x;
	}

MatrixXd stackedAutoEncoder::extractFeature(MatrixXd x) {
	MatrixXd prelayer_x(x);
	for (int i = 0; i < hidden.size(); i++) {
		prelayer_x = layer[i].predict(prelayer_x, 1, 0);
		}
	return prelayer_x;
	}

MatrixXd stackedAutoEncoder::reconstruct(MatrixXd x) {
	MatrixXd prelayer_x;
	prelayer_x = extractFeature(x);
	for (int i = hidden.size()-1; i >=0; i--) {
		prelayer_x = layer[i].predict(prelayer_x, 1, 1);
		}
	return prelayer_x;
	}
