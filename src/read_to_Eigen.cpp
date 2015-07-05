#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
using namespace std;

namespace Eigen {
	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	void read(Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &m, const char* filename) {
		std::ifstream f(filename);
		for(int i = 0; i < m.rows(); i++) {
			for(int j = 0; j < m.cols(); j++){
				f >> m(i, j);
				}
			}
		f.close();
		}

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	void operator<< (Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &m, const ifstream f) {
		for(int i = 0; i < m.rows(); i++) {
			for(int j = 0; j < m.cols(); j++){
				f >> m(i, j);
				}
			}
		}
	}

using namespace Eigen;

/* test
int main() {
	ifstream f("unif.txt");
	ofstream  out("matrix.txt");
	MatrixXd m(4, 4);

	//read(m, "unif.txt");
	m << f;
	f.close();

	out << m << endl;
	out.close();

	return 0;	
	}
*/
