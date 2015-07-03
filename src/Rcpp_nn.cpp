List C_feedForwardNetwork(MatrixXd x, Matrix y,
	VectorXi size, string actf = "sigm", string out = "sigm",
	int numepochs = 3, int batchsize = 100, method = "bp",
	VectorXd param, double hid_dropout, double vis_dropout, bool verbose = false
	) {

	feedForwardNetwork nn(size, actf, out, hid_dropout, vis_dropout);
	VectorXd error;

	if (method = "bp") {
		error = nn.bpTrain(x, y, numepochs, batchsize, param[0], param[1], param[2], verbose);
	} else {
		error = nn.rpropTrain(x, y, numepochs, batchsize, param[0], param[1], param[2], param[3],  verbose);
		}

	return List::create(
		Named("network") = nn,
		Named("error") error
		);
		}
	

List C_feedForwardNetwork_train_more(feedForwardNetwork nn, MatrixXd x, Matrix y,
	int numepochs = 3, int batchsize = 100, method = "bp",
	VectorXd param, double hid_dropout, double vis_dropout, bool verbose = false
	) {

	VectorXd error;
	if (method = "bp") {
		error = nn.bpTrain(x, y, numepochs, batchsize, param[0], param[1], param[2], verbose);
	} else {
		error = nn.rpropTrain(x, y, numepochs, batchsize, param[0], param[1], param[2], param[3],  verbose);
		}

	return List::create(
		Named("network") = nn,
		Named("error") error
		);
	}
