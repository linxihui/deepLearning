#include <RcppCommon.h>
#include "deepnet.hpp"


namespace Rcpp {
	template <> SEXP wrap(const RBM &);
	template <> RBM as(SEXP);

	template <> SEXP wrap(const feedForwardNetwork &);
	template <> feedForwardNetwork as(SEXP);

	template <> SEXP wrap(const stackedAutoEncoder &);
	template <> stackedAutoEncoder as(SEXP);
	}

#include <Rcpp.h>


namespace Rcpp {
	template <> SEXP wrap(const RBM & rbm) {
		return List::create(
			Named("W") = rbm.W,
			Named("vBias") = rbm.vBias,
			Named("hBias") = rbm.hBias,
			Named("delta_W") = rbm.delta_W,
			Named("delta_vBias") = rbm.delta_vBias,
			Named("delta_hBias") = rbm.delta_hBias,
			Named("size") = rbm.size
			);
		}

	template <> RBM as(SEXP rbm_R) {
		List rbm_L(rbm_R);
		MatrixXd W(rbm_L["W"]), delta_W(rbm_L["delta_W"]);
		VectorXd vBias(rbm_L["vBias"]), delta_vBias(rbm_L["delta_vBias"]);
		VectorXd hBias(rbm_L["hBias"]), delta_hBias(rbm_L["delta_hBias"]);
		Vector2i size(rbm_L["size"]);

		RBM rbm(W, delta_W, vBias, delta_vBias, hBias, delta_vBias, size[0], size[1]);

		return rbm;
		}


	template <> SEXP wrap(const feedForwardNetwork & nn) {
		return List::create(
			Named("layer_size") = nn.layer_size,
			Named("act_fun") = nn.act_fun,
			Named("output") = nn.output,
			Named("hidden_dropout") = nn.hidden_dropout,
			Named("visible_dropout") = nn.visible_dropout,
			Named("W") = nn.W,
			Named("B") = nn.B
			);
		}

	template <> feedForwardNetwork as(SEXP nn_R) {
		List nn_L(nn_R);
		VectorXi layer_size(nn_L["layer_size"]);
		string act_fun(nn_L["act_fun"]);
		string output(nn_N["output"]);
		double hidden_dropout(nn_N["hidden_dropout"]); 
		double visible_dropout(nn_N["visible_dropout"]);
		List W(nn_R["W"]);
		List B(nn_R["B"]);

		feedForwardNetwork nn(layer_size, act_fun, output, hidden_dropout, visible_dropout);
		for (int i = 1; i < W.size(); i++) {
			nn.W[i] = as<MatriXd>(W[i]);
			nn.B[i] = as<VectorXd>(B[i]);
			}

		return nn;
		}

		int input_dim;
		VectorXi hidden;
		vector<feedForwardNetwork> layer;

	template <> SEXP wrap(const stackedAutoEncoder & sae) {
		return List::create(
			Named("input_dim") = sae.input_dim,
			Named("hidden") = sae.hidden,
			Named("layer") = layer
			);
		}

	template <> stackedAutoEncoder as(SEXP sae_R) {
		List sae_L(sae_R);
		int input_dim(sae_L["input_dim"]);
		VectorXi hidden(sae_L["hidden"]);
		List layer_L(sae_R["layer"]);

		stackedAutoEncoder sae(input_dim, hidden);
				
		for (int i = 0; i < layer_L.size(); i++) {
			sae.layer[i] = as<feedForwardNetwork>(layer_L[i]);
			}

		return sae;
		}
	}
