/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;

REGISTER_OP("SingleImageRandomDotStereograms")
    .Attr("T: {double,float,int64,int32}")   
    .Input("depth_values: T")
    .Output("image: uint8")
    .Attr("hidden_surface_removal: bool = true")
    .Attr("convergence_dots_size: int = 8")
    .Attr("dots_per_inch: int = 72")
    .Attr("eye_separation: float = 2.5")
    .Attr("mu: float = .3333")
    .Attr("normalize: bool = true")
    .Attr("normalize_max: float = -100.0")
    .Attr("normalize_min: float = 100.0")
    .Attr("boarder_level: float = 0.0")
    .Attr("number_colors: int = 256")    
    .Attr("generation_mode: string = 'SIRDS'")
    .Attr("output_image_shape: shape = { dim {size:1024} dim {size: 768} dim {size: 1}}")
    .Attr("output_data_window: shape = { dim {size:1022} dim {size: 757}}")
    // TODO(Mazecreator):  Add "SetShapeFN" to reflect proper output shape & Rank 2 error checking on Input
    // TODO(Mazecreator):  Add better error handling for output shapes / windows settings
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //   ::tensorflow::shape_inference::ShapeHandle input;
    //   ::tensorflow::shape_inference::ShapeHandle output;
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &output));
    //   c->set_output(0,output);
    //   return Status::OK();
    // })
    .Doc(R"doc(
Output a RandomDotStereogram Tensor of shape "output_image_shape" for export via encode_PNG or encode_JPG OP.

Based upon:
'http://www.learningace.com/doc/4331582/b6ab058d1e206d68ab60e4e1ead2fe6e/sirds-paper'

Example use which outputs a SIRDS image as picture_out.png:
img=[[1,2,3,3,2,1],
     [1,2,3,4,5,2],
     [1,2,3,4,5,3],
     [1,2,3,4,5,4],
     [6,5,4,4,5,5]]

session = tf.InteractiveSession()

sirds = single_image_random_dot_stereograms(img,convergence_dots_size=8,number_colors=256,normalize=True)

out = sirds.eval()

png = tf.image.encode_png(out).eval()

with open('picture_out.png', 'wb') as f:
    f.write(png)


depth_values:           Z values of data to encode into "output_data_window" window, lower further away {0.0 floor(far), 1.0 ceiling(near) after normalization}, must be rank 2
hidden_surface_removal: Activate hidden surface removal (True)
convergence_dots_size:  Black dot size in pixels to help view converge image, drawn on bottom of image (8 pixels)
dots_per_inch:	        Output device in dots/inch (72 default)
eye_separation:         Separation between eyes in inches (2.5 inchs)
mu:                     Depth of field, Fraction of viewing distance (1/3 = .3333)
normalize:              Normalize input data to [0.0, 1.0] (True)
normalize_max:          Fix MAX value for Normalization (0.0) - if < MIN, autoscale
normalize_min:          Fix MIN value for Normalization (0.0) - if > MAX, autoscale
boarder_level:          Value of board in depth 0.0 {far} to 1.0 {near} (0.0)
number_colors:          2 (Black & White),256 (grayscale), and Numbers > 256 (Full Color) are all that are supported currently
generation_mode:        Mode for Stereogram
                            SIRDS - 2 color stereogram (Default)
output_image_shape:     Output size of returned image in X,Y, Channels 1-grayscale, 3 color (1024, 768, 1), channels will be updated to 3 if number_colors > 256
output_data_window:     Size of "DATA" window, must be equal to or smaller than output_image_shape, will be centered
                          and use convergence_dots_size for best fit to avoid overlap if possible

image:                  returns a Tensor of size output_image_shape with depth_values encoded into image

)doc");
