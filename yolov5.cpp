// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolov5.h"
#include <iostream>

#include <float.h>
#include <cpu.h>
#include <simpleocv.h>

using namespace std; 

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    printf("Generate purposals");
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = 2;

    printf("Starting to generate purposals for %d anchors", num_anchors);
    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        printf("Anchor %f %f", anchor_w, anchor_h);
        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = feat_blob.channel(q * 85 + 5 + k).row(i)[j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * 85 + 4).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * 85 + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * 85 + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * 85 + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * 85 + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

YOLOv5::YOLOv5()
{
}

int YOLOv5::load()
{
    yolov5.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    yolov5.opt.num_threads = ncnn::get_big_cpu_count();
    // yolov5.load_param("best-220601-640-normal.param");
    // yolov5.load_model("best-220601-640-normal.bin");
    yolov5.load_param("best-220526-opt-fp16-640.param");
    yolov5.load_model("best-220526-opt-fp16-640.bin");

    // yolov5.load_param("best-dynamic.param");
    // yolov5.load_model("best-dynamic.bin");

    // yolov5.load_param("best-normal (1)-opt.param");
    // yolov5.load_model("best-normal (1)-opt.bin");

    return 0;
}

void prettyPrint(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

void transpose(ncnn::Mat& in, ncnn:: Mat& out) {
    int w = in.w; // 1
    int h = in.h; // 640
    int d = in.d; // 640
    int channels = in.c; // 3

    out.create(w, channels, h, d, in.elemsize);

    for (int q = 0; q < d; q++) {
        float* outptr = out.channel(q);
        for (int z = 0; z < h; z++) {
            for (int i = 0; i < channels; i++) {
                const float* ptr = in.channel(i).depth(q).row(z);
                for (int j = 0; j < w; j++) {
                    *outptr++ = ptr[j];
                }
            }
        }
    }
}

int YOLOv5::detect(const cv::Mat& rgba, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    printf("detect\n");
    int width = rgba.cols;
    int height = rgba.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(rgba.data, ncnn::Mat::PIXEL_RGB, width, height);

    // prettyPrint(in);

    // printf("Color (%d, %d, %d)\n", in.shape().w, in.shape().h, in.shape().c);

    in = in.reshape(1, 3, 640, 640);

    // // prettyPrint(in);

    // printf("Reshape (%d, %d, %d, %d)\n", in.shape().w, in.shape().h, in.shape().d, in.shape().c);

    // const float norm_vals[4] = {1 / 255.f, 1 / 255.f, 1 / 255.f, 1 / 255.f};
    // in.substract_mean_normalize(0, norm_vals);

    // printf("Normalized (%d, %d, %d, %d)\n", in.shape().w, in.shape().h, in.shape().d, in.shape().c);

    // ncnn::Mat transposed = ncnn::Mat();

    // transpose(in, transposed);

    // ncnn::Mat shape = transposed.shape();
    // printf("Transposed (%d, %d, %d, %d)\n", shape.w, shape.h, shape.d, shape.c);

    ncnn::Extractor ex = yolov5.create_extractor();

    ex.input("images", in);
    printf("Input %d\n", in.dims);
    
    ncnn::Mat out;
    int res = ex.extract("output", out);

    printf("---------------------- Output %d (res %d)\n", out.dims, res);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // // stride 8
    // {

    //     ncnn::Mat out;
    //     ex.extract("output", out);

    //     ncnn::Mat anchors(6);
    //     anchors[0] = 10.f;
    //     anchors[1] = 13.f;
    //     anchors[2] = 16.f;
    //     anchors[3] = 30.f;
    //     anchors[4] = 33.f;
    //     anchors[5] = 23.f;

    //     std::vector<Object> objects8;
    //     generate_proposals(anchors, 8, in, out, prob_threshold, objects8);

    //     memset(text1, 0, sizeof(text1));
    //     sprintf(text1, "Stride 8 %lu", objects8.size());
    //     print_log(text1);

    //     proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    // }

    // // stride 16
    // {
    //     memset(text1, 0, sizeof(text1));
    //     sprintf(text1, "Stride 16");
    //     print_log(text1);

    //     ncnn::Mat out;
    //     ex.extract("353", out);

    //     ncnn::Mat anchors(6);
    //     anchors[0] = 30.f;
    //     anchors[1] = 61.f;
    //     anchors[2] = 62.f;
    //     anchors[3] = 45.f;
    //     anchors[4] = 59.f;
    //     anchors[5] = 119.f;

    //     std::vector<Object> objects16;
    //     generate_proposals(anchors, 16, in, out, prob_threshold, objects16);

    //     proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    // }

    // // stride 32
    // {

    //     ncnn::Mat out;
    //     ex.extract("367", out);

    //     ncnn::Mat anchors(6);
    //     anchors[0] = 116.f;
    //     anchors[1] = 90.f;
    //     anchors[2] = 156.f;
    //     anchors[3] = 198.f;
    //     anchors[4] = 373.f;
    //     anchors[5] = 326.f;

    //     std::vector<Object> objects32;
    //     generate_proposals(anchors, 32, in, out, prob_threshold, objects32);

    //     proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    // }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    printf("NMSed\n");

    int count = picked.size();

    printf("received %d objects\n", count);

    objects.resize(count);

    // pad to target_size rectangle
    int wpad = 0;
    int hpad = 0;
    int scale = 1;

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int YOLOv5::draw(cv::Mat& rgba, const std::vector<Object>& objects)
{
    static const char* class_names[] = { "dick", "dick-head" };

    static const unsigned char colors[3][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2], 255);

        cv::rectangle(rgba, cv::Rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height), cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgba.cols)
            x = rgba.cols - label_size.width;

        cv::rectangle(rgba, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0, 255) : cv::Scalar(255, 255, 255, 255);

        cv::putText(rgba, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }

    return 0;
}
