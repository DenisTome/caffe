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
  // top[0]->Reshape(bottom[0]->num(), channels_, height_, width_); <-- It should be the same
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
	if (max_y == -1){
		// Input has zero value
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
	Mat M;
	hconcat(Mat(posX),Mat(posY),M);
	Mat elem = Mat(elemsList) / sumVals;
	Mat elem_r = repeat(elem,1,2);
	Mat M_t;
	transpose(M, M_t);
	Mat p1;
	caffe_mul(M.rows*M.cols, (float*)M.data, (float*)elem_r.data, (float*)p1.data);// M.mul(elem_r);
//	Mat C = M_t *
	LOG(INFO) << "M shape: (" << M.rows << "," << M.cols << ")";
//	LOG(INFO) << "C shape: (" << C.rows << "," << C.cols << ")";
//    for (int t=0;t<18;t++){
//        cout << "M val (" << t << ",0):" << M.at<float>(t,1) << endl;
//    }
//    cout << "C is:["<<C.at<float>(0)<<","<<C.at<float>(1)<<","<<C.at<float>(2)<<","<<C.at<float>(3)<<"]"<<endl;
//	memcpy(&(cov[0]), C.data, C.rows*C.cols*sizeof(float));
	// TODO: substitute operations with caffe functions (like caffe_mul)
	// TODO: convert caffe operations in gpu (caffe_gpu_operation)
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

//  int channelOffset = width_ * height_;
  // scan batches
//  for (int b = 0; b < bottom[0]->num(); b++){
//	  // scan channels
//	  int batch_offset = b * channelOffset * bottom[0]->num();
//	  for (int hm = 0; hm < njoints_; hm++){
//		  int curr_offset = batch_offset + hm * channelOffset;
//	  }
//  }
  Point2f mean;
  Vec4f cov;
  fitGaussian(bottom_data, mean, cov);

}

template <typename Dtype>
void ManifoldLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  if (!propagate_down[0]) {
//    return;
//  }
//  const Dtype* top_diff = top[0]->cpu_diff();
//  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//  // Different pooling methods. We explicitly do the switch outside the for
//  // loop to save time, although this results in more codes.
//  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
//  // We'll output the mask to top[1] if it's of size >1.
//  const bool use_top_mask = top.size() > 1;
//  const int* mask = NULL;  // suppress warnings about uninitialized variables
//  const Dtype* top_mask = NULL;
//  switch (this->layer_param_.pooling_param().pool()) {
//  case PoolingParameter_PoolMethod_MAX:
//    // The main loop
//    if (use_top_mask) {
//      top_mask = top[1]->cpu_data();
//    } else {
//      mask = max_idx_.cpu_data();
//    }
//    for (int n = 0; n < top[0]->num(); ++n) {
//      for (int c = 0; c < channels_; ++c) {
//        for (int ph = 0; ph < pooled_height_; ++ph) {
//          for (int pw = 0; pw < pooled_width_; ++pw) {
//            const int index = ph * pooled_width_ + pw;
//            const int bottom_index =
//                use_top_mask ? top_mask[index] : mask[index];
//            bottom_diff[bottom_index] += top_diff[index];
//          }
//        }
//        bottom_diff += bottom[0]->offset(0, 1);
//        top_diff += top[0]->offset(0, 1);
//        if (use_top_mask) {
//          top_mask += top[0]->offset(0, 1);
//        } else {
//          mask += top[0]->offset(0, 1);
//        }
//      }
//    }
//    break;
//  case PoolingParameter_PoolMethod_AVE:
//    // The main loop
//    for (int n = 0; n < top[0]->num(); ++n) {
//      for (int c = 0; c < channels_; ++c) {
//        for (int ph = 0; ph < pooled_height_; ++ph) {
//          for (int pw = 0; pw < pooled_width_; ++pw) {
//            int hstart = ph * stride_h_ - pad_h_;
//            int wstart = pw * stride_w_ - pad_w_;
//            int hend = min(hstart + kernel_h_, height_ + pad_h_);
//            int wend = min(wstart + kernel_w_, width_ + pad_w_);
//            int pool_size = (hend - hstart) * (wend - wstart);
//            hstart = max(hstart, 0);
//            wstart = max(wstart, 0);
//            hend = min(hend, height_);
//            wend = min(wend, width_);
//            for (int h = hstart; h < hend; ++h) {
//              for (int w = wstart; w < wend; ++w) {
//                bottom_diff[h * width_ + w] +=
//                  top_diff[ph * pooled_width_ + pw] / pool_size;
//              }
//            }
//          }
//        }
//        // offset
//        bottom_diff += bottom[0]->offset(0, 1);
//        top_diff += top[0]->offset(0, 1);
//      }
//    }
//    break;
//  case PoolingParameter_PoolMethod_STOCHASTIC:
//    NOT_IMPLEMENTED;
//    break;
//  default:
//    LOG(FATAL) << "Unknown pooling method.";
//  }
	return;
}


#ifdef CPU_ONLY
STUB_GPU(ManifoldLayer);
#endif

INSTANTIATE_CLASS(ManifoldLayer);

}  // namespace caffe
