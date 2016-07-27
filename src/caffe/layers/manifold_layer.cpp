#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>

#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ManifoldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ManifoldParameter manifold_param = this->layer_param_.manifold_param();
  njoints_ = manifold_param.njoints();
  channels_ = manifold_param.njoints() + 1;
  sigma_ = manifold_param.sigma();
  debug_mode_ = manifold_param.debug_mode();
  max_area_ = manifold_param.max_area();
  percentage_max_ = manifold_param.percentage_max();
}

template <typename Dtype>
void ManifoldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->channels() == channels_)
	  << "Number of heat-maps does not match with the number of joints.";
  width_ = bottom[0]->height();
  height_ = bottom[0]->width();
  CHECK(width_ == height_) << "Input expected to have the same width and height";

  top[0]->ReshapeLike(*bottom[0]);
  //top[0]->Reshape(bottom[0]->num(), channels_, height_, width_); //<-- It should be the same
}

template<typename Dtype>
void ManifoldLayer<Dtype>::putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma){
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1)
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template<typename Dtype>
void ManifoldLayer<Dtype>::fitGaussian(const Dtype* data, Point2f &mean, Vec4f &cov){
	int max_y = -1;
	int max_x = -1;
	float maxVal = -1.0;
	// get mean
	for (int g_y = 0; g_y < height_; g_y++){
	  for (int g_x = 0; g_x < width_; g_x++){
		  if (data[g_y*width_ + g_x] > maxVal){
			  maxVal = data[g_y*width_ + g_x];
			  max_y = g_y;
			  max_x = g_x;
		  }
	  }
	}
	if (maxVal <= 0.0){
		// Input has zero value
		// Return joint in the middle of the heat-map with maximum uncertainty
		mean = Point2f((width_/2),(height_/2));
		cov = Vec4f(width_*height_, 0, 0, width_*height_);
		return;
	}
	mean = Point2f(max_x,max_y);

	// get convariance matrix
	double sumVals = 0.0;
	vector<float> elemsList;
	vector<float> posX;
	vector<float> posY;
	int numElements = 0;
	for (int g_y = 0; g_y < height_; g_y++){
	  for (int g_x = 0; g_x < width_; g_x++){
		  if ((data[g_y*width_ + g_x] > (maxVal*percentage_max_*0.01)) &&
			  (sqrt(pow(g_y-mean.y,2)+pow(g_x-mean.x,2)) < max_area_)){
			  numElements++;
			  sumVals += data[g_y*width_ + g_x];
			  elemsList.push_back(data[g_y*width_ + g_x]);
			  posX.push_back(g_x - mean.x);
			  posY.push_back(g_y - mean.y);
		  }
	  }
	}
	Mat M(posX.size(),2,CV_32FC1);
	if (debug_mode_){
		LOG(INFO) << "posX size " << posX.size();
		LOG(INFO) << "posY size " << posY.size();
		LOG(INFO) << "elem size " << elemsList.size();
	}
	CHECK(posX.size() == posY.size()) << "Number of elements in estimating the covariance matrix must be the same";
	CHECK(posX.size() == elemsList.size()) << "Number of weights in estimating the covariance matrix must be the same";
	hconcat(Mat(posX),Mat(posY),M);
	Mat elem = repeat(Mat(elemsList) / sumVals,1,2);
	Mat M_t;
	transpose(M, M_t);
	Mat p1(M.rows, M.cols, CV_32FC1);
	caffe_mul(M.rows*M.cols, (float*)M.data, (float*)elem.data, (float*)p1.data);
	Mat C = M_t * p1;
	memcpy(&(cov[0]), C.data, C.rows*C.cols*sizeof(float));
	// TODO: convert caffe operations in gpu (caffe_gpu_operation)
}

template <typename Dtype>
void ManifoldLayer<Dtype>::findNewJointPositions(Point2f* means, Vec4f* cov, Point2f* newPoints){
	// TODO: do actual manifold conversion
	for (int j=0;j<njoints_;j++){
		newPoints[j] = means[j];
	}
}

// TODO: finish
template <typename Dtype>
void ManifoldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int top_count = top[0]->count();

	// Initialising all heat-maps to zero
	caffe_set(top_count, Dtype(0), top_data);

	int channelOffset = width_ * height_;
	int batch_offset = channelOffset * channels_;
	// Scan batches
	if (debug_mode_)
		LOG(INFO) << "BATCH size is " << bottom[0]->num();
	int b = 0;
	//	for (int b = 0; b < bottom[0]->num(); b++){
		// Scan channels
		if (debug_mode_)
			LOG(INFO) << "considering batch image " << b;
		Point2f mean[njoints_];
		Vec4f cov[njoints_];
		for (int hm = 0; hm < njoints_; hm++){
			int curr_offset = b * batch_offset + hm * channelOffset;
			fitGaussian(bottom_data+curr_offset, mean[hm], cov[hm]);
		}

		if (debug_mode_)
			LOG(INFO) << "manifold get new joints ";
		// TODO
	    // Extracted mean and convariance are input to the manifold layer where these data
	    // are used to identify the new 2D joint coordinates.
		Point2f newJointPos[njoints_];
		findNewJointPositions(&mean[0], &cov[0], &newJointPos[0]);

		if (debug_mode_)
			LOG(INFO) << "generate new heat-maps";
		// Generate heat-map with new joint positions
		for (int hm = 0; hm < njoints_; hm++){
			int curr_offset = b * batch_offset + hm * channelOffset;
			putGaussianMaps(top_data+curr_offset, newJointPos[hm], 1, width_, height_, sigma_);
		}

		for (int g_y = 0; g_y < height_; g_y++){
			for (int g_x = 0; g_x < width_; g_x++){
				float maximum = 0;
				for (int i = 0; i < njoints_; i++){
					maximum = (maximum > top_data[b*batch_offset + i*channelOffset + g_y*width_ + g_x]) ? maximum : top_data[b*batch_offset + i*channelOffset + g_y*width_ + g_x];
				}
				top_data[b*batch_offset + njoints_*channelOffset + g_y*width_ + g_x] = max(1.0-maximum, 0.0);
			}
		}

		if (debug_mode_){
			for (int j = 0; j < channels_; j++){
				int curr_offset = b * batch_offset + j * channelOffset;
				char imagename [100];
				sprintf(imagename, "manifold_batch_%02d_before_after_%02d.jpg", b, j);
				int padding = 2;

				Mat tmp_vis_in(height_,width_,CV_32FC1);
				Mat tmp_vis_out(height_,width_,CV_32FC1);
				Mat before_after = Mat::ones(height_+2*padding,2*width_+3*padding,CV_32FC1);
				memcpy(tmp_vis_in.data,bottom_data+curr_offset, width_*height_*sizeof(float));
				memcpy(tmp_vis_out.data,top_data+curr_offset, width_*height_*sizeof(float));

				Mat before_after_vis;
				tmp_vis_in.copyTo(before_after(Rect(padding,padding,height_,width_)));
				tmp_vis_out.copyTo(before_after(Rect(width_+2*padding,padding,height_,width_)));
				before_after.convertTo(before_after_vis, CV_8UC1, 255, 0);
				imwrite(imagename, before_after_vis);
			}
		}
//	}

}

template <typename Dtype>
void ManifoldLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set(top[0]->count(), Dtype(0), bottom_diff);
	return;
}


#ifdef CPU_ONLY
STUB_GPU(ManifoldLayer);
#endif

INSTANTIATE_CLASS(ManifoldLayer);

}  // namespace caffe
