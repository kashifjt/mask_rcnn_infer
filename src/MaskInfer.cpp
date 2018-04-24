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
  auto vec = ConvertImageToVector(image);
  Tensor input_image(DT_FLOAT, TensorShape({1,image.rows,image.cols,image.channels()}));
  copy_n(vec.begin(), vec.size(), input_image.flat<float>().data());
  Tensor input_image_meta(DT_FLOAT, TensorShape({1,image_meta.rows,image_meta.cols,image_meta.channels()}));
  auto vec = ConvertImageToVector(image_meta);
  Tensor input_image_meta(DT_FLOAT, TensorShape({1,image_meta.rows,image_meta.cols,image_meta.channels()}));

  // StringPiece tmp_input_image_data = input_image.tensor_data();
  // memcpy(const_cast<char*>(tmp_input_image_data.data()), (image.data), image.rows * image.cols * image.channels() * sizeof(float));
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
  string output_layer = "output_mrcnn_class";
  std::vector<Tensor> outputs;
  Status run_status = tensorflow_session->Run({{in_image_nm, input_image}},
                                               //{in_image_meta_nm, input_image_meta},
                                               //{in_image_anchors_nm, input_image_anchors}},
                                               {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
     LOG(ERROR) << "Running model failed: " << run_status;
     }
  return outputs;
}
