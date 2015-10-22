#include "deepnet.hpp"

// feedForwardNetwork::feedForwardNetwork(feedForwardNetwork &nn) {
// 	layer_size = nn.layer_size;
// 	act_fun = nn.act_fun;
// 	output = nn.output;
// 	hidden_dropout = nn.hidden_dropout;
// 	visible_dropout = nn.visible_dropout;
// 	W = nn.W;
// 	B = nn.B;
// 	post.resize(layer_size.size());
// 	dropout_mask.resize(layer_size.size()-1);
// 	}


feedForwardNetwork::feedForwardNetwork(stackedAutoEncoder &sae, int output_dim, string out, double hid_dropout, double vis_dropout) {
	layer_size.setConstant(sae.hidden.size() + 2, -1);
	layer_size[0] = sae.input_dim;
	layer_size.segment(1, sae.hidden.size()) = sae.hidden;
	layer_size[sae.hidden.size() + 1] = output_dim;
	act_fun = sae.layer[0].act_fun;
	output = out;
	hidden_dropout = hid_dropout;
	visible_dropout = vis_dropout;
	post.resize(layer_size.size());
	dropout_mask.resize(layer_size.size()-1);
	//
	W.resize(layer_size.size()-1);
	B.resize(layer_size.size()-1);
	
	int i = 0; 
	for(;i < layer_size.size()-2; i++) {
		W[i] = sae.layer[i].W[0];
		B[i] = sae.layer[i].B[0];
		}
	W[i] = W[i].Random(layer_size[i+1], layer_size[i])*0.1;
	B[i] = B[i].Random(layer_size[i+1])*0.1;
	}


			
feedForwardNetwork::feedForwardNetwork(DBN & dbn, int output_dim, string out, double hid_dropout, double vis_dropout) {
	layer_size.setConstant(dbn.hidden.size() + 2, -1);
	layer_size[0] = dbn.input_dim;
	layer_size.segment(1, dbn.hidden.size()) = dbn.hidden;
	layer_size[dbn.hidden.size() + 1] = output_dim;
	act_fun = "sigm";
	output = out;
	hidden_dropout = hid_dropout;
	visible_dropout = vis_dropout;
	post.resize(layer_size.size());
	dropout_mask.resize(layer_size.size()-1);
	//
	W.resize(layer_size.size()-1);
	B.resize(layer_size.size()-1);
	
	int i = 0; 
	for(;i < layer_size.size()-2; i++) {
		W[i] = dbn.layer[i].W[0];
		B[i] = dbn.layer[i].B[0];
		}
	W[i] = W[i].Random(layer_size[i+1], layer_size[i])*0.1;
	B[i] = B[i].Random(layer_size[i+1])*0.1;
	}


feedForwardNetwork::feedForwardNetwork(VectorXi size, string actf, string out, double hid_dropout, double vis_dropout) {
	layer_size = size;
	act_fun = actf;
	output = out;
	hidden_dropout = hid_dropout;
	visible_dropout = vis_dropout;
	W.resize(layer_size.size()-1);
	B.resize(layer_size.size()-1);
	post.resize(layer_size.size());
	dropout_mask.resize(layer_size.size()-1);
	for(int i = 0; i < layer_size.size()-1; i++) {
		W[i] = W[i].Random(layer_size[i+1], layer_size[i])*0.1;
		B[i] = B[i].Random(layer_size[i+1])*0.1;
		}
	}


MatrixXd feedForwardNetwork::ff(MatrixXd batch_x, MatrixXd batch_y) {
	return batch_y - ff(batch_x);
	}

MatrixXd feedForwardNetwork::ff(MatrixXd batch_x) {
	//x: row = feature, col = sample
	//m : sample
	if (layer_size.size() > 2) {
		ff(batch_x, layer_size.size() - 2);
		}
	// output layer
	int i = layer_size.size() - 1;

	MatrixXd error;
	post[i].noalias() = W[i-1]*post[i-1];
	post[i].colwise() += B[i-1];
	if (output == "sigm") {
		post[i] = sigm(post[i]);
	} else if (output == "softmax") {
		post[i] = post[i].array().exp();
		//post[i] = post[i].array() / post[i].colwise().sum().replicate(layer_size[i], 1).array();
		post[i] = post[i].array().colwise() / post[i].colwise().sum().transpose().array();
		}
	return post[i];
	}

MatrixXd feedForwardNetwork::ff(MatrixXd batch_x, int nstep) {
	//x: row = feature, col = sample
	//m : sample
	int n_sample = batch_x.cols();

	if (visible_dropout > 0) {
		dropout_mask[0] = dropoutMask(batch_x.rows(), visible_dropout);
		batch_x = batch_x.array() * dropout_mask[0].replicate(1, n_sample).array();
		}
	post[0] = batch_x;
		
	int i = 1;
	MatrixXd pre;
	for (; i <= nstep; i++) {
		//pre = W[i-1]*post[i-1] + B[i-1].replicate(1, n_sample);
		pre.noalias() = W[i-1]*post[i-1];
		pre.colwise() += B[i-1];
		if (act_fun == "sigm") {
			post[i] = sigm(pre);
			}
		else if (act_fun == "tanh") {
			post[i] = tanh(pre);
			}
		else if (act_fun == "rect") {
			post[i] = pre.array().max(0);
			}
		else if (act_fun == "srect") {
			post[i] = (1 + pre.array().exp()).log();
			}
		if(hidden_dropout > 0) {
			dropout_mask[i] = dropoutMask(post[i].rows(), hidden_dropout);
			//post[i] = post[i].array() * dropout_mask[i].replicate(1, n_sample).array() ;
			//post[i].array().colwise() *= dropout_mask[i].array();
			post[i] = (dropout_mask[i].array() > 0).select(post[i], 0);
			}
		}
	return post[nstep];
	}


void feedForwardNetwork::errorBackPropFromOutput(MatrixXd & d) {
	// d: by sample error
	int layer = layer_size.size() - 1;
	if (output == "sigm") {
		d = -d.array() * (post[layer].array() * (1. - post[layer].array()));
	} else if (output == "linear" || output == "softmax") {
		d = -d;
		}
	}


void feedForwardNetwork::errorBackProp(MatrixXd & d, int i) {
	MatrixXd d_act; // MatrixXd::Zero(d.rows(), d.cols());
	if (act_fun  == "tanh" ) {
		d_act = 1 - post[i].array().square();
		}
	else if (act_fun == "rect") {
		d_act = (post[i].array() > 0).cast<double>();
		}
	else if (act_fun == "srect") {
		d_act = 1 - (-post[i]).array().exp();
	} else { //act_fun  == "sigm") 
		d_act  = post[i].array() * (1-post[i].array());
		} 
	d = (W[i].transpose()*d).array() * d_act.array();
	if	(hidden_dropout > 0) {
		d = d.array() * dropout_mask[i].replicate(1, d.cols()).array();
		}
	}


MatrixXd feedForwardNetwork::predict(MatrixXd batch_x) {
	//x: row = feature, col = sample
	return predict(batch_x, layer_size.size() - 1, 0);	
	}

MatrixXd feedForwardNetwork::predict(MatrixXd batch_x, int nstep, int start) {
	//x: row = feature, col = sample
	MatrixXd pred(batch_x);
	int end = (nstep + start <= layer_size.size()-2) ? nstep + start : layer_size.size()-2;
	int i = start;
	while(i < end) {
		pred.noalias() = W[i]*pred;
		pred.colwise() += B[i];
		if (act_fun == "sigm") {
			pred = sigm(pred);
			}
		else if (act_fun == "tanh") {
			pred = tanh(pred);
			}
		else if (act_fun == "rect") {
			pred = pred.array().max(0);
			}
		else if (act_fun == "srect") {
			pred = (1 + pred.array().exp()).log();
			}
		i++;
		}
	if ( i < start + nstep) {
		// output layer
		//i = layer_size.size() - 2;
		MatrixXd error;
		pred.noalias() = W[i]*pred;
		pred.colwise() += B[i];
		if (output == "sigm") {
			pred = sigm(pred);
		} else if (output == "softmax") {
			pred = pred.array().exp();
			pred = pred.array().colwise() / pred.colwise().sum().transpose().array();
			}
		}
	return pred;
	}


// MatrixXd feedForwardNetwork::ff(MatrixXd batch_x, MatrixXd batch_y) {
// 	//x: row = feature, col = sample
// 	//m : sample
// 	int n_sample = batch_x.cols();
// 
// 	if (visible_dropout > 0) {
// 		dropout_mask[0] = dropoutMask(batch_x.rows(), visible_dropout);
// 		batch_x = batch_x.array() * dropout_mask[0].replicate(1, n_sample).array();
// 		}
// 	post[0] = batch_x;
// 		
// 	int i = 1;
// 	MatrixXd pre;
// 	for (; i < layer_size.size() - 1; i++) {
// 		//pre = W[i-1]*post[i-1] + B[i-1].replicate(1, n_sample);
// 		pre.noalias() = W[i-1]*post[i-1];
// 		pre.colwise() += B[i-1];
// 		if (act_fun == "sigm") {
// 			post[i] = sigm(pre);
// 			}
// 		else if (act_fun == "tanh") {
// 			post[i] = tanh(pre);
// 			}
// 		else if (act_fun == "rect") {
// 			post[i] = pre.array().max(0);
// 			}
// 		else if (act_fun == "srect") {
// 			post[i] = (1 + pre.array().exp()).log();
// 			}
// 		if(hidden_dropout > 0) {
// 			dropout_mask[i] = dropoutMask(post[i].rows(), hidden_dropout);
// 			//post[i] = post[i].array() * dropout_mask[i].replicate(1, n_sample).array() ;
// 			//post[i].array().colwise() *= dropout_mask[i].array();
// 			post[i] = (dropout_mask[i].array() > 0).select(post[i], 0);
// 			}
// 		}
// 	// output layer
// 	MatrixXd error;
// 	post[i].noalias() = W[i-1]*post[i-1];
// 	post[i].colwise() += B[i-1];
// 	if (output == "sigm") {
// 		post[i] = sigm(post[i]);
// 	} else if (output == "softmax") {
// 		post[i] = post[i].array().exp();
// 		//post[i] = post[i].array() / post[i].colwise().sum().replicate(layer_size[i], 1).array();
// 		post[i] = post[i].array().colwise() / post[i].colwise().sum().transpose().array();
// 		}
// 	error = batch_y - post[i];
// 	return error;
// 	}
// 
