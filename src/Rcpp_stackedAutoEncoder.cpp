List C_stackedAutoEncoder(MatrixXd x,
	VectorXi hidden, string act_fun = "sigm", string reconst_fun = "sigm",
	int numepochs = 10, int batchsize = 100, method = "bp",
	VectorXd param, double hidden_dropout, double visual_dropout, bool verbose = false
	) {

	stackedAutoEncoder sae(x.rows(), hidden, act_fun, reconst_fun, hidden_dropout, visual_dropout);
	MatrixXd x_reconst;

	if (method = "bp") {
		x_reconst = sae.bpTrain(x, numepochs, batchsize, param[0], param[1], param[2], verbose);
	} else {
		x_reconst =sae.rpropTrain(x, numepochs, batchsize, param[0], param[1], param[2], param[3],  verbose);
		}

	return List::create(
		Named("stackedAutoEncoder") = sae,
		Named("reconstruction") =  x_reconst
		);
	}
	

List C_stackedAutoEncoder_train_more(stackedAutoEncoder sae, MatrixXd x,
	int numepochs = 10, int batchsize = 100, method = "bp",
	VectorXd param, double hid_dropout, double vis_dropout, bool verbose = false
	) {

	MatrixXd x_reconst;

	if (method = "bp") {
		x_reconst = sae.bpTrain(x, numepochs, batchsize, param[0], param[1], param[2], verbose);
	} else {
		x_reconst = sae.rpropTrain(x, numepochs, batchsize, param[0], param[1], param[2], param[3],  verbose);
		}

	return List::create(
		Named("stackedAutoEncoder") = sae,
		Named("reconstruction") =  x_reconst
		);
	}
