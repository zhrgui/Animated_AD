import numpy as np
import json

def create_timeline(segments, t_start, t_end, step=0.01):
    """
    Create a time grid from t_start to t_end (inclusive) and assign a speaker label
    for each time point if the time falls within a segment.
    If a time point is not covered by any segment, the label is None.
    Assumes segments do not overlap.
    """
    times = np.arange(t_start, t_end + step, step)
    labels = np.array([None] * len(times), dtype=object)
    for seg in segments:
        seg_start, seg_end, label = seg['start'], seg['end'], seg['label']
        indices = np.where((times >= seg_start) & (times < seg_end))[0]
        # If segments do overlap, later segments will override earlier ones.
        # For DER calculation including overlaps we will use a separate count.
        labels[indices] = label
    return times, labels

def create_ignore_mask(segments, times, collar=0.25):
    """
    For each ground truth segment, create a boolean mask for times that fall within
    the collar region at the segment's boundaries.
    """
    ignore = np.zeros_like(times, dtype=bool)
    for seg in segments:
        seg_start, seg_end = seg['start'], seg['end']
        # Collar region at beginning
        start_ignore = (times >= seg_start) & (times < seg_start + collar)
        ignore = ignore | start_ignore
        # Collar region at end
        end_ignore = (times >= seg_end - collar) & (times < seg_end)
        ignore = ignore | end_ignore
    return ignore

def create_overlap_mask(segments, times):
    """
    Create a mask that marks time points where more than one ground truth segment is active,
    i.e. overlapping speech.
    """
    count = np.zeros(len(times))
    for seg in segments:
        indices = np.where((times >= seg['start']) & (times < seg['end']))[0]
        count[indices] += 1
    # Overlap if count is greater than 1.
    return count > 1

def compute_der_for_video(gt_segments, pred_segments, collar=0.25, step=0.01, ignore_other=True):
    """
    Computes DER for one video and returns:
      - error_time_incl: Error time including overlapping regions.
      - total_gt_time_incl: Total ground truth speech time (excluding collar regions) including overlaps.
      - error_time_excl: Error time computed only on non-overlapping regions.
      - total_gt_time_excl: Total ground truth speech time (excluding collar regions) that is non-overlapping.
    
    If ignore_other is True, any time points covered by a ground truth segment labeled "Other"
    are ignored so that predictions during those times do not count as false alarms.
    """
    # Determine timeline boundaries using all segments.
    t_start = min(seg['start'] for seg in gt_segments)
    t_end = max(seg['end'] for seg in gt_segments)
    
    # Create timeline for ground truth and predictions.
    times, full_gt_labels = create_timeline(gt_segments, t_start, t_end, step)
    _, pred_labels = create_timeline(pred_segments, t_start, t_end, step)
    
    # Create collar mask from ground truth segments.
    ignore_collar = create_ignore_mask(gt_segments, times, collar)
    
    # Choose segments for overlap computation.
    if ignore_other:
        segments_for_overlap = [seg for seg in gt_segments if seg['label'] != "Other"]
    else:
        segments_for_overlap = gt_segments
    overlap_mask = create_overlap_mask(segments_for_overlap, times)
    
    # Initialize accumulators.
    error_time_incl = 0.0
    total_gt_time_incl = 0.0
    error_time_excl = 0.0
    total_gt_time_excl = 0.0
    
    for i in range(len(times)):
        gt_label = full_gt_labels[i]
        pred_label = pred_labels[i]
        
        # If ignoring "Other," skip these timestamps entirely so
        # predictions during "Other" do not count as false alarms.
        if ignore_other and gt_label == "Other":
            continue
        
        in_collar = ignore_collar[i]
        
        # Determine if the time point is a ground-truth speech label (not "Other" or None).
        is_gt_speech = (gt_label is not None) and (gt_label != "Other")
        
        # Count ground truth speech time only if it is labeled as speech and not in collar.
        if is_gt_speech and not in_collar:
            total_gt_time_incl += step
            if not overlap_mask[i]:
                total_gt_time_excl += step

        # Check for errors: misses (ground truth speech, but prediction doesn’t match)
        # or false alarms (ground truth is silence/None, but a speech prediction is made).
        if not in_collar:
            if is_gt_speech:
                if pred_label != gt_label:
                    error_time_incl += step
                    if not overlap_mask[i]:
                        error_time_excl += step
            else:
                # For non-speech ground truth, a prediction is a false alarm.
                if pred_label is not None:
                    error_time_incl += step
                    if not overlap_mask[i]:
                        error_time_excl += step

    return error_time_incl, total_gt_time_incl, error_time_excl, total_gt_time_excl

def compute_overall_der(gt_dict, pred_dict, collar=0.25, step=0.01, ignore_other=True):
    """
    Computes global DER over multiple videos for both:
      - All regions (including overlapping speech)
      - Only non-overlapping speech regions.
      
    The ignore_other flag determines whether time points with a ground truth label "Other"
    are ignored in the evaluation.
    
    Returns:
      overall_der_incl: DER including overlapping speech.
      overall_der_excl: DER computed only on non-overlapping regions.
    """
    total_error_time_incl = 0.0
    total_gt_time_incl = 0.0
    total_error_time_excl = 0.0
    total_gt_time_excl = 0.0

    # Process each video.
    for video_id, video_gt_segments in gt_dict.items():
        # If ignoring "Other", skip videos with no valid (non-"Other") ground truth speech.
        if ignore_other:
            valid = any(seg['label'] != "Other" for seg in video_gt_segments)
        else:
            valid = len(video_gt_segments) > 0
        
        if not valid:
            continue

        video_pred_segments = pred_dict.get(video_id, [])
        et_incl, gt_incl, et_excl, gt_excl = compute_der_for_video(
            video_gt_segments, video_pred_segments, collar, step, ignore_other
        )
        total_error_time_incl += et_incl
        total_gt_time_incl += gt_incl
        total_error_time_excl += et_excl
        total_gt_time_excl += gt_excl

    if total_gt_time_incl == 0 or total_gt_time_excl == 0:
        raise ValueError("Total ground truth speech time is zero (after collar and 'Other' exclusion).")
    
    overall_der_incl = total_error_time_incl / total_gt_time_incl
    overall_der_excl = total_error_time_excl / total_gt_time_excl

    return overall_der_incl, overall_der_excl

if __name__ == "__main__":
    gt_dict = {}
    pred_dict = {}

    gt_file = "/users/zhongrui/avobjects_zgui/eval/cmd_test_closed_set_annotations_der.json"
    pred_file = "/scratch/shared/beegfs/zhongrui/autoad_data/results/0405_diarisation/subtitling_results_visual_correction.json"

    with open(gt_file, 'r') as infile:
        gt_annotations = json.load(infile)

    with open(pred_file, 'r') as infile:
        predictions = json.load(infile)
    
    ignore_other = False

    # Process ground truth annotations
    for movie_title, movie_annotations in gt_annotations.items():
        for clip_idx, clip_annotations in movie_annotations.items():
            instances = []
            for gt_instance in clip_annotations:
                instances.append({
                    "start": gt_instance["temporal_range"][0],
                    "end": gt_instance["temporal_range"][1],
                    "label": gt_instance["character"]
                })
            gt_dict[clip_idx] = instances

    # Process prediction annotations
    for movie_title, movie_predictions in predictions.items():
        for clip_idx, clip_predictions in movie_predictions.items():

            instances = []
            for pred_instance in clip_predictions:

                words = pred_instance["text"].split()
                word_count = len(words)
                duration = pred_instance["temporal_range"][1] - pred_instance["temporal_range"][0]

                if duration / word_count >= 1.5:
                    continue

                if ignore_other:
                    instances.append({
                        "start": pred_instance["temporal_range"][0],
                        "end": pred_instance["temporal_range"][1],
                        "label": pred_instance["label"]
                    })
                else:
                    if pred_instance["score"] <= 0.15:
                        label = "Other"
                    else:
                        label = pred_instance["label"]

                    instances.append({
                        "start": pred_instance["temporal_range"][0],
                        "end": pred_instance["temporal_range"][1],
                        "label": label
                    })

            pred_dict[clip_idx] = instances

    # Set ignore_other to True or False to switch versions.
    overall_der_incl, overall_der_excl = compute_overall_der(
        gt_dict, pred_dict, collar=0.25, step=0.01, ignore_other=ignore_other
    )
    print("Overall DER including overlapping speech: {:.2%}".format(overall_der_incl))
    print("Overall DER on non-overlapping speech: {:.2%}".format(overall_der_excl))
