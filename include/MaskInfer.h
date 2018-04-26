#ifndef MASKINFER_H
#define MASKINFER_H

#include <fstream>
#include <utility>
#include <vector>
#include <string>
#include "Config.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace tensorflow;
using namespace std;

class MaskInfer
{
private:
  vector<string> labels;
  int labels_count;
  Config config;
  GraphDef tensorflow_graph;
  unique_ptr<Session> tensorflow_session;

public:
  MaskInfer(Config myconfig);
  Status ReadLabelsFile(const string& file_name);
  Status LoadGraph(const string& file_name);
  vector<Tensor> infer(Mat& image, Mat& image_meta, Mat& image_anchors);
  Status PrintLabels(Tensor& classTensor);
  Mat tensor_to_cvmat(Tensor& inTensor, int MatType=CV_32FC1, bool IgnoreFirstDim = false, bool LastDimChannel=false);
};
#endif
