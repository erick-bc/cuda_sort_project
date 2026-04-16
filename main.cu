#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>

using std::vector;
using std::cout;

int main() {
  	// we want to sort something on device
  	thrust::device_vector<int> P = {224, 238, 664, 943, 349, 378, 872, 706, 786, 956, 686, 833, 420, 335, 343, 428, 930, 108,
                   212, 159, 67, 278, 297, 660, 70, 248, 415, 793, 64, 165, 672, 532, 173, 674, 903, 459, 268,
                   36, 430, 452, 767, 439, 349, 126, 688, 397, 890, 833, 602, 757, 556, 743, 901, 238, 674, 640,
                   677, 463, 80, 355, 419, 1, 851, 590, 938, 419, 841, 648, 61, 640, 421, 267, 783, 263, 262, 658, 
                   328, 974, 179, 797, 206, 258, 949, 737, 525, 459, 138, 405, 795, 355, 712, 872, 786, 481, 861, 
                   845, 943, 732, 432, 961};

	thrust::sort(P.begin(), P.end());

	for (int i = 1; i < P.size(); i++) {
		cout << P[i - 1] << ", ";

		if (i % 15 == 0) {
			cout << std::endl;
		}
	}

	cout << std::endl;

	return 0;
}