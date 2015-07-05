rbm <- function(x, hidden, numepochs, batchsize, learning_rate, learning_rate_scale, momentum, cd) {
	.Call("C_RBM", x, hidden, numepochs, batchsize, learning_rate, learning_rate_scale, momentum, cd, PACKAGE = 'deepLearning')
	}

rbm.more(rbm, x, learning_rate, learning_rate_scale, momentum, cd) {
	.Call("C_RBM_train_more", x, learning_rate, learning_rate_scale, momentum, cd, PACKAGE = 'deepLearning')
	}
