List C_RBM(MatrixXd x, int hidden, int numepochs = 3, int batchsize = 100, 
	double learning_rate = 0.8, double learning_rate_scale = 1, int cd = 1,
	double momentum = 0.5
	) {
	RBM rbm(xx.cols(), hidden);
	VectorXd error = rbm.train(x, numepochs, batchsize, learning_rate, momentum, learning_rate_scale, cd);
	return List::create(
		Named("rbm") = wrap(rbm),
		Named("error") = error
		);
	}


List C_RBM_train_more(RBM rbm, Matrix x, int numepochs = 3, int batchsize = 100, 
	double learning_rate, double momentum, double learning_rate_scale, int cd
	) {
	//RBM rbm(rbm_trained);
	VectorXd error = rbm.train(x, numepochs, batchsize, learning_rate, momentum, learning_rate_scale, cd);
	return List::create(
		Named("rbm") = wrap(rbm),
		Named("error") = error
		);
	}
