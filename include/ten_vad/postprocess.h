#pragma once
#include <vector>
#include "infer.h"


struct Segment {
  float start;
  float end;
  int label;
};

std::vector<Segment> ExtractSegments(
    const std::vector<VadResult>& results,
    float frame_ms = 16.0f);

std::vector<Segment> ExtractSegmentsWithPostProcess(
    const std::vector<VadResult>& results,
    float frame_ms = 16.0f);

std::vector<Segment> PostProcessSegments(
    const std::vector<Segment>& input);
