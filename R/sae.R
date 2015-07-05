#' @title Train Restricted Boltzmann Machine
#' @param x Input matrix: row = sample, col = feature
#' @param hidden A vector of numbers of hidden neurons
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

sae <- function(x, hidden, act_fun = 'sigm', out_fun = 'sigm', method = c('bp', 'rprop'), param,
	numepochs = 10L, batchsize = 100L, dropout = c(0, 0), verbose = FALSE) {
	method <- match.arg(method);
	param.default <- switch(method,
		'bp' = list(learning_rate = 0.1, momentum = 0.5, learning_rate_scale),
		'rprop' = list(inc_scale = 1.2, dec_scale = 0.5, inc_scale_max = 50, dec_scale_max = 1e-6)
		);
	param <- modifyList(param.default, param);
	x <- t(x);
	out <- .Call("C_stackedAutoEncoder", x, hidden, act_fun, out_fun, numepochs, batchsize, 
		method, unlist(param), dropout[2], dropout[1], verbose, PACKAGE = 'deepLearning');
	out$act_fun <- act_fun;
	out$out_fun <- out_fun;
	out$numepochs <- numepochs;
	out$batchsize <- batchsize;
	out$train <- c(list(method = method), param);
	return(structure(out, class = 'SAE'));
	}

#' @rdname sae
#' @export
sae.more <- function(sae, x, numepochs = 10L, batchsize = 100L, 
	method = sae$train$method, param = sae$train$param, verbose = FALSE) {
	x <- t(x);
	temp <- .Call("C_stackedAutoEncoder_train_more", sae$stackedAutoEncoder, x, numepochs, batchsize, 
		method = method, unlist(param), verbose, PACKAGE = 'deepLearning');
	sae <- modifyList(sae, temp);
	sae$numepochs <- c(sae$numepochs, numepochs);
	sae$batchsize <- c(sae$batchsize, batchsize);
	return(sae);
	}
