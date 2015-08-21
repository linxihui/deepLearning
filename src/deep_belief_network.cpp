class DBN {
	public:
		int input_dim;
		VectorXi hidden;
		vector<feedForwardNetwork> layer;

		DBN(int input_dim, VectorXi hid);
		//
		std::vector<VectorXd> train(MatrixXd &x, double learning_rate = 0.1, 
			double learning_rate_scale = 1, double momentum = 0.5, int numepochs = 10, 
			int batchsize = 100, int CD = 1, bool verbose = true);
		//
		//MatrixXd move_down(MatrixXd h, int round = 10);
		MatrixXd extractHiddenFeature(MatrixXd x);
		MatrixXd reconstructInput(MatrixXd x);
	};


DBN::DBN(int in_dim, VectorXi hid) {
	input_dim = in_dim;
	hidden = hid;
	VectorXi vis_hid(hidden.size() + 1);
	vis_hid[0] = input_dim;
	vis_hid.segment(1, hidden.size()) = hidden;
	layer.reserve(hidden.size());
	for (int i = 0; i < hidden.size(); i++) {
		layer.push_back(rbm(vis_hid[i], vis_hid[i+1]));
		}
	}


MatrixXd DNB::train(MatrixXd &x, double learning_rate, 
	double learning_rate_scale, double momentum, int numepochs, 
	int batchsize = 100, int CD = 1, bool verbose = true) {
	MatrixXd prelayer_x(x);
	for (int i = 0; i < hidden.size(); i++) {
		if (verbose) cout << "Training layer " << i+1 << "..." << endl;
		layer[i].train(prelayer_x, numepochs, batchsize, learning_rate, momentum, learning_rate_scale, CD);
		prelayer_x = layer[i].prob_h_given_v(prelayer_x)
		}
	return prelayer_x;
	}



MatrixXd DBN::extractFeature(MatrixXd x) {
	MatrixXd prelayer_x(x);
	for (int i = 0; i < hidden.size(); i++) {
		prelayer_x = layer[i].prob_h_given_v(prelayer_x);
		}
	return prelayer_x;
	}

MatrixXd DBN::reconstruct(MatrixXd x) {
	MatrixXd prelayer_x;
	prelayer_x = extractFeature(x);
	for (int i = hidden.size()-1; i >=0; i--) {
		prelayer_x = layer[i].prob_h_given_v(prelayer_x);
		}
	return prelayer_x;
	}
