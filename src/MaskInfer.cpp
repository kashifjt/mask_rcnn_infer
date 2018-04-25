#include "MaskInfer.h"

MaskInfer::MaskInfer(Config myconfig)
{
  config = myconfig;
}

Status MaskInfer::ReadLabelsFile(const string& file_name) {
  ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  labels.clear();
  string line;
  while (std::getline(file, line)) {
    labels.push_back(line);
  }
  labels_count = labels.size();
  if (config.NUM_CLASSES != labels_count) {
    return tensorflow::errors::InvalidArgument("Labels count does not match.");
  }
  return Status::OK();
}

void MaskInfer::SetConfig(Config config){
  this->config = config;
}

Status MaskInfer::LoadGraph(const string& file_name) {
    Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), file_name, &tensorflow_graph);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", file_name, "'");
    }
    tensorflow_session.reset(NewSession(SessionOptions()));
    Status session_create_status = tensorflow_session->Create(tensorflow_graph);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

std::vector<float> ConvertImageToVector(Mat& image) {
  std::vector<float> image_value;
  if (image.isContinuous()) {
    image_value.assign(
              (float*)image.datastart,
              (float*)image.dataend);
  }
  else {
   for (int i = 0; i < image.rows; i++) {
     image_value.insert(
             image_value.end(),
             image.ptr<float>(i),
             image.ptr<float>(i) + image.cols);
    }
  }
  return image_value;
 }

vector<Tensor> MaskInfer::infer(Mat& image, Mat& image_meta, Mat& image_anchors)
{
  cout<<"image shape: "<<image.size()<<endl;
  cout<<"image_meta shape: "<<image_meta.size()<<endl;
  cout<<"image_anchors shape: "<<image_anchors.size()<<endl;

  Tensor input_image(DT_FLOAT, TensorShape({1,image.rows,image.cols,image.channels()}));
  float * i = input_image.flat<float>().data();
  Mat fake_image(image.rows, image.cols, CV_32FC3, i);
  image.convertTo(fake_image, CV_32FC3);

  Tensor input_image_meta(DT_FLOAT, TensorShape({1,image_meta.cols}));
  float * m = input_image_meta.flat<float>().data();
  Mat fake_image_meta(image_meta.rows, image_meta.cols, CV_32FC1, m);
  image_meta.convertTo(fake_image_meta, CV_32FC1);

  Tensor input_image_anchors(DT_FLOAT, TensorShape({1,image_anchors.rows,image_anchors.cols}));
  float * a = input_image_anchors.flat<float>().data();
  Mat fake_image_anchors(image_anchors.rows, image_anchors.cols, CV_32FC1, a);
  image_anchors.convertTo(fake_image_anchors, CV_32FC1);
//  vector<float> vec_image = ConvertImageToVector(image);
//  vector<float> vec_image_meta = ConvertImageToVector(image_meta);
//  vector<float> vec_image_anchors = ConvertImageToVector(image_anchors);
//  Tensor input_image(DT_FLOAT, TensorShape({1,image.rows,image.cols,image.channels()}));
//  copy_n(vec_image.begin(), vec_image.size(), input_image.flat<float>().data());
//  Tensor input_image_meta(DT_FLOAT, TensorShape({1,image_meta.rows,image_meta.cols,image_meta.channels()}));
//  copy_n(vec_image_meta.begin(), vec_image_meta.size(), input_image_meta.flat<float>().data());
//  Tensor input_image_anchors(DT_FLOAT, TensorShape({1,image_anchors.rows,image_anchors.cols,image_anchors.channels()}));
//  copy_n(vec_image_anchors.begin(), vec_image_anchors.size(), input_image_anchors.flat<float>().data());
//
//  StringPiece tmp_input_image_data = input_image.tensor_data();
//  memcpy(const_cast<char*>(tmp_input_image_data.data()), (image.data), image.rows * image.cols * image.channels() * sizeof(float));
  // Tensor input_image_meta(DT_FLOAT, TensorShape({1,image_meta.rows,image_meta.cols,image_meta.channels()}));
  // StringPiece tmp_input_image_meta_data = input_image_meta.tensor_data();
  // memcpy(const_cast<char*>(tmp_input_image_meta_data.data()), (image_meta.data), image_meta.rows * image_meta.cols * sizeof(float));
  // Tensor input_image_anchors(DT_FLOAT, TensorShape({1,image_anchors.rows,image_anchors.cols,image_anchors.channels()}));
  // StringPiece tmp_input_image_anchors_data = input_image_anchors.tensor_data();
  // memcpy(const_cast<char*>(tmp_input_image_anchors_data.data()), (image_anchors.data), image_anchors.rows * image_anchors.cols * sizeof(float));
  // auto root = tensorflow::Scope::NewRootScope();
  // auto sh = tensorflow::ops::Shape(root, input_image);
//   int height = image.rows;
//   int width = image.cols;
//   int depth = image.channels();
//
//   Tensor input_image(DT_FLOAT, TensorShape({1,image.rows,image.cols,image.channels()}));
//   auto input_tensor_mapped = input_image.tensor<float, 4>();
//   auto source_data = image.data;
//   for (int y = 0; y < height; ++y) {
//     const float* source_row = source_data + (y * width * depth);
//     for (int x = 0; x < width; ++x) {
//         const float* source_pixel = source_row + (x * depth);
//         for (int c = 0; c < depth; ++c) {
//            const float* source_value = source_pixel + c;
//            input_tensor_mapped(0, y, x, c) = *source_value;
//         }
//     }
// }

  string in_image_nm = "input_image_1";
  string in_image_meta_nm = "input_image_meta_1";
  string in_image_anchors_nm = "input_anchors_1";
  string output_layer_detection = "output_detections";
  string output_layer_class = "output_mrcnn_class";
  string output_layer_bbox = "output_mrcnn_bbox";
  string output_layer_mask = "output_mrcnn_mask";
  vector<Tensor> outputs;
  Status run_status = tensorflow_session->Run({{in_image_nm, input_image},
                                               {in_image_meta_nm, input_image_meta},
                                               {in_image_anchors_nm, input_image_anchors}},
                                               {output_layer_detection, output_layer_class,
                                                output_layer_bbox, output_layer_mask}, {}, &outputs);
  if (!run_status.ok()) {
     LOG(ERROR) << "Running model failed: " << run_status;
     }
  return outputs;
}

Status MaskInfer::PrintLabels(Tensor& classTensor)
{
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_labels";
TopK(root.WithOpName(output_name), classTensor, 1);
GraphDef graph;
TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

std::unique_ptr<tensorflow::Session> session(
    tensorflow::NewSession(tensorflow::SessionOptions()));
TF_RETURN_IF_ERROR(session->Create(graph));
// The TopK node returns two outputs, the scores and their original indices,
// so we have to append :0 and :1 to specify them both.
std::vector<Tensor> out_tensors;
TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                {}, &out_tensors));
Tensor scores = out_tensors[0];
Tensor indices = out_tensors[1];

tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
for (int pos = 0; pos < 1000; ++pos) {
  const int label_index = indices_flat(pos);
  const float score = scores_flat(pos);
  if(indices_flat(pos) != 0)
  cout << labels[label_index] << " (" << label_index << "): " << score<<endl;
}
return Status::OK();
}

void MaskInfer::tensor_to_cvmat(Tensor& inTensor, int MatType, bool IgnoreFirstDim, bool LastDimChannel)
{
    vector<int> dim_size;
    for (int i = 0; i<inTensor.shape().dims(); ++i)
    {
        if(IgnoreFirstDim && i ==0)
            continue;
        dim_size.push_back(inTensor.dim_size(i));
    }
    int dims_count = (LastDimChannel) ? dim_size.size() - 1: dim_size.size();
    cout<<"dims_count: "<<dims_count<<endl;
    Mat mat(dims_count, &dim_size[0], MatType);
    cout<<"Size: "<<mat.size()<<endl;
    cout<<"DType: "<<inTensor.dtype()<<endl;
    memcpy(mat.data, inTensor.tensor_data().data(), inTensor.tensor_data().size());
    cout<<mat.row(0)<<endl;

}
