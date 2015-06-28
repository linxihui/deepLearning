#include "deepnet.hpp"

MatrixXd sigm(MatrixXd x) {
	return ((-x.array()).exp() + 1.0).inverse().matrix();
	}

MatrixXd tanh(MatrixXd x) {
	ArrayXXd out = (2.*x).array().exp();
	return ((out - 1) / (out + 1)).matrix();
	}

MatrixXd bernoulli_sample(MatrixXd prob) {
	MatrixXd out = MatrixXd::Random(prob.rows(), prob.cols());
	return (prob.array() > out.array().abs()).cast<double>().matrix();
	}

VectorXd dropoutMask(int size, double fraction) {
	VectorXd out = VectorXd::Random(size);
	return (out.array().abs() > fraction).cast<double>();
	}

MatrixXd extractCols(MatrixXd &x, VectorXd indices) {
	MatrixXd subx(x.rows(), indices.size());
	for(int i = 0; i < indices.size(); i++) 
		subx.col(i) = x.col(indices(i));
	return subx;
	}

int updateCols(MatrixXd &x, MatrixXd &up, VectorXd indices) {
	for(int i = 0; i < indices.size(); i++) 
		x.col(indices(i)) = up.col(i);
	return 1;
	}

ArrayXXi sign(ArrayXXd x) {
	return ((x >= 0).cast<int>() - (x <= 0).cast<int>());
	}
