#' @title Train Restricted Boltzmann Machine
#' @param x Input matrix: row = sample, col = feature
#' @param hidden Number of hidden neurons
#' @param numepochs Number of epoches
#' @param batchsize Min-batch size
#' @param learning_rate Learning speed / shrinkage
#' @param momentum Fraction of last update to researve
#' @param Learning_rate_scale Scalar of of learning_rate
#' @param cd Number of steps of contrast divergence
#' @param rbm RBM object
#' @return RBM object
#' @details `rbm` trains RBM object; `rbm.more` trains more epoches.
#' @export
rbm <- function(x, hidden, numepochs = 10L, batchsize = 100L, 
	learning_rate = 0.1, momentum = 0.5, learning_rate_scale = 1.0, cd = 1L) {
	x <- t(x);
	out <- .Call("C_RBM", x, hidden, numepochs, batchsize, learning_rate, 
		momentum, learning_rate_scale, cd, PACKAGE = 'deepLearning');
	out$numepochs <- numepochs;
	out$batchsize <- batchsize;
	out$learning_rate <- learning_rate;
	out$learning_rate_scale <- learning_rate_scale;
	out$cd <- cd;
	return(structure(out, class = 'RBM'));
	}

#' @rdname rbm
#' @export
rbm.more <- function(rbm, x, numepochs = 10L, batchsize = 100L, 
	learning_rate = rbm$learning_rate, momentum = rbm$momentum, 
	learning_rate_scale = rbm$learning_rate_scale, cd = rbm$cd) {
	x <- t(x);
	out <- .Call("C_RBM_train_more", rbm$rbm, x, numepochs, batchsize, 
		momentum, learning_rate_scale, cd, PACKAGE = 'deepLearning');
	out$error <- c(rbm$error, out$error);
	out$numepochs <- c(rbm$numepochs, numepochs);
	out$batchsize <- c(rbm$batchsize, batchsize);
	out$learning_rate <- c(rbm$learning_rate, learning_rate);
	out$learning_rate_scale <- c(rbm$learning_rate_scale, learning_rate_scale);
	out$cd <- c(rbm$cd, cd);
	return(structure(out, class = 'RBM'));
	}
