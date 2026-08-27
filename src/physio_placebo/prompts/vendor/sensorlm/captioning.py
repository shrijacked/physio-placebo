"""Creating Caption for Sensor-LM data."""

import datetime
import random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from .constants import (
    low_level_templates, activity_templates, trend_templates, 
    anomaly_templates, mood_templates_with_time, 
    NORMALIZATION_PARAMETERS, FEATURES_TO_NORMALIZE, 
    channel_groups_selected, channel_groups_random, random_choose_nums
)

def generate_statistical_caption(
    x_raw: np.ndarray, mask: np.ndarray, caption_with_impute_vals: bool = True
) -> str:
  """Generate a caption describing the mean (or sum) of each channel, grouped by category.

  Args:
      x_raw: A np.ndarray of shape [len(features), N], where N is the time
        length, and features are the name of features of the channels. Note that
        the input tensort representes the raw data, not the normalized data.
      mask: A np.ndarray of the same shape as x, where 1 indicates missingness.
      caption_with_impute_vals: If True, imputed values will be used in the
        caption.

  Returns:
      A formatted string describing each channel, grouped by category.
  """

  labels = FEATURES_TO_NORMALIZE
  norm_params = NORMALIZATION_PARAMETERS

  if x_raw.shape[1] != len(labels):
    raise ValueError(
        f'Input tensor must have {len(labels)} channels, but got'
        f' {x_raw.shape[1]}.'
    )

  if mask is not None and not caption_with_impute_vals:
    if mask.shape != x_raw.shape:
      raise ValueError(
          f'Mask shape {mask.shape} must match input shape {x_raw.shape}.'
      )
    # set imputed values to NaN for exclusion in computations
    x_masked = np.where(mask == 1, np.nan, x_raw)
  else:
    x_masked = x_raw

  # denormalize the sensor values (ignore mask for now)
  original_x = denormalize_sensor_values(x_masked, labels, norm_params)

  # calculate mean, max, min, std on denormalized data
  channel_stats = []
  for i in range(len(labels)):
    channel_data = original_x[:, i]
    channel_stats.append({
        'mean': np.nanmean(channel_data),
        'max': np.nanmax(channel_data),
        'min': np.nanmin(channel_data),
        'std': np.nanstd(channel_data),
    })

  feature_stats_dict = dict(zip(labels, channel_stats))

  caption_parts = []
  for category, group in channel_groups_selected.items():
    category_parts = []
    for name, member in zip(group['names'], group['members']):
      stats = feature_stats_dict.get(member, {})
      if stats:
        mean_val = stats.get('mean', np.nan)
        max_val = stats.get('max', np.nan)
        min_val = stats.get('min', np.nan)
        std_val = stats.get('std', np.nan)

        if all(
            not np.isnan(val) for val in [mean_val, max_val, min_val, std_val]
        ):
          category_parts.append(
              _describe_low_level(name, mean_val, max_val, min_val, std_val)
          )

    # add random features
    group_random = channel_groups_random.get(category)
    if group_random:
      for _ in range(random_choose_nums[category]):
        random_index = random.randrange(len(group_random['names']))
        name = group_random['names'][random_index]
        member = group_random['members'][random_index]
        stats = feature_stats_dict.get(member, {})
        if stats:
          mean_val = stats.get('mean', np.nan)
          max_val = stats.get('max', np.nan)
          min_val = stats.get('min', np.nan)
          std_val = stats.get('std', np.nan)

          if all(
              not np.isnan(val) for val in [mean_val, max_val, min_val, std_val]
          ):
            category_parts.append(
                _describe_low_level(name, mean_val, max_val, min_val, std_val)
            )

    if category_parts:
      caption_parts.append(
          f'For {category}, ' + ' '.join(category_parts) + '\n'
      )

  caption = ''.join(caption_parts)
  return caption


def generate_structural_caption(
    x_raw: np.ndarray, max_num_insight_per_category: int = 7
) -> str:
  """Generate a mid level caption describing the events in the data."""
  labels = FEATURES_TO_NORMALIZE
  norm_params = NORMALIZATION_PARAMETERS

  sensor_data = denormalize_sensor_values(x_raw, labels, norm_params)
  sensor_data = _average_downsample(sensor_data.T, 36)

  df = pd.DataFrame(sensor_data.T, columns=FEATURES_TO_NORMALIZE)
  df = _update_minute_index(df, 40)

  data = {}
  caption = ''

  for col in df.columns:
    data[col] = list(df[col])

  for category, group in channel_groups_selected.items():
    insights = []

    for name, member in zip(group['names'], group['members']):

      sensor_name = name

      regression_trends = _identify_trends_with_regression(
          data[member], downsample_scale=40
      )
      regression_trends = sorted(
          regression_trends, key=lambda item: abs(item[0]), reverse=False
      )
      for start, end, trend_type, _, _, _ in regression_trends:
        insights.append(_describe_trend(sensor_name, trend_type, start, end))

      peak_and_valleys = _detect_significant_peaks_valleys(data[member])
      for time, anomaly in peak_and_valleys:
        insights.append(_describe_anomaly(sensor_name, anomaly, time))
    if len(insights) > max_num_insight_per_category:
      insights = [
          insights[i]
          for i in sorted(
              random.sample(range(len(insights)), max_num_insight_per_category)
          )
      ]
    caption += f"{category}: {' '.join(insights)}\n"

  return caption


def generate_semantic_caption(
    x_raw: np.ndarray,
    mask: np.ndarray,
    activity_name_list: List[str],
    starttime_list: list[Tuple[str, int]],
    endtime_list: list[Tuple[str, int]],
    sleep_starttime_list: List[str],
    sleep_endtime_list: List[str],
    top_k_activity: int = 8,
    top_k_sleep: int = 2,
    mood_data_list: List[str] | None = None,
    mood_timestamp_list: List[str] | None = None,
    min_activity_duration_minutes: int = 20,
) -> Tuple[str, str]:
  """Generate a high level caption describing the events in the data.

  Args:
      x_raw: A np.ndarray of shape [len(features), N], where N is the time
        length, and features are the name of features of the channels. Note that
        the input tensort representes the raw data, not the normalized data.
      mask: A np.ndarray of the same shape as x, where 1 indicates missingness.
      activity_name_list: A list of activity names.
      starttime_list: A list of start time strings.
      endtime_list: A list of end time strings.
      sleep_starttime_list: A list of sleep start time strings.
      sleep_endtime_list: A list of sleep end time strings.
      top_k_activity: The number of top activities to include in the caption.
      top_k_sleep: The number of top sleep periods to include in the caption.
      mood_data_list: A list of mood data.
      mood_timestamp_list: A list of mood timestamps.
      min_activity_duration_minutes: The minimum duration of an activity to be
        included in the caption.

  Returns:
      A formatted string describing each channel, grouped by category.
  """

  caption_activity, caption_sleep = '', ''

  event_lists = []

  activity_list = []
  for activity, (start_time, start_minute), (end_time, end_minute) in zip(
      activity_name_list, starttime_list, endtime_list
  ):
    duration = end_minute - start_minute + 1
    activity_list.append((activity, start_minute, end_minute, duration))

  # sort by duration and get top k = 8
  sorted_activity_list = sorted(activity_list, key=lambda x: x[3])
  for i, activity_event in enumerate(sorted_activity_list):
    activity, start_minute, end_minute, duration = activity_event
    if duration < min_activity_duration_minutes:
      continue
    caption_activity = (
        caption_activity
        + _describe_activity(activity, start_minute, end_minute)
        + ' '
    )
    event_lists.append((activity, start_minute, end_minute))
    if len(event_lists) >= top_k_activity:
      break

  sleep_list = []
  if sleep_starttime_list and sleep_endtime_list:
    for start_time, end_time in zip(sleep_starttime_list, sleep_endtime_list):
      start_minute = _get_minute_of_day(start_time)
      end_minute = _get_minute_of_day(end_time)
      sleep_list.append((start_minute, end_minute, end_minute - start_minute))

    # sort by duration and get top k = 2
    sorted_sleep_list = sorted(sleep_list, key=lambda x: x[2])

    for i, sleep_event in enumerate(sorted_sleep_list):
      if i >= top_k_sleep:
        break
      start_minute, end_minute, _ = sleep_event
      caption_sleep = (
          caption_sleep
          + _describe_activity('Sleep', start_minute, end_minute)
          + ' '
      )
      event_lists.append(('Sleep', start_minute, end_minute))

  caption_events = '\n'.join([caption_activity, caption_sleep])

  for mood, time in zip(mood_data_list, mood_timestamp_list):
    mood_minute = _get_minute_of_day(time[:19] + '+00:00')
    caption_events = (
        caption_events + '\n' + _describe_mood(mood, mood_minute) + '\n\n'
    )

  return caption_events


def _describe_mood(mood, time) -> str:
  """Selects a random template and formats it with the provided trend information."""
  template = random.choice(mood_templates_with_time)
  return template.format(mood=mood, time=time)


def _describe_trend(sensor_name, trend_type, start_minute, end_minute) -> str:
  """Selects a random template and formats it with the provided trend information."""
  template = random.choice(trend_templates)
  return template.format(
      sensor_name=sensor_name,
      trend_type=trend_type,
      start=start_minute,
      end=end_minute,
  )


def _describe_anomaly(sensor_name, anomaly, time) -> str:
  """Selects a random template and formats it with the provided anomaly information."""
  template = random.choice(anomaly_templates)
  return template.format(sensor_name=sensor_name, anomaly=anomaly, time=time)


def _describe_low_level(
    sensor_name, mean_val, max_val, min_val, std_val
) -> str:
  """Selects a random template and formats it with the provided trend information."""
  template = random.choice(low_level_templates)
  return template.format(
      name=sensor_name,
      mean_val=mean_val,
      max_val=max_val,
      min_val=min_val,
      std_val=std_val,
  )


def _describe_activity(activity, start_minute, end_minute) -> str:
  """Selects a random template and formats it with the provided trend information."""
  template = random.choice(activity_templates)
  return template.format(
      activity=activity, start_minute=start_minute, end_minute=end_minute
  )


def denormalize_sensor_values(
    x: np.ndarray,
    labels: List[str],
    norm_params: Dict[str, list[float]],
) -> np.ndarray:
  """Denormalizes sensor values.

  Args:
      x: A np.ndarray of shape [len(features), N], where N is the time length,
        and features are the name of features of the channels.
      labels: A list of feature labels.
      norm_params: A dictionary of normalization parameters.

  Returns:
      The denormalized sensor values.
  """

  # denormalize the sensor values
  original_x = np.zeros_like(x)
  for i in range(len(labels)):
    original_x[:, i] = (
        x[:, i] * norm_params[labels[i]][1] + norm_params[labels[i]][0]
    )

  # make 'steps' and 'sleep_coefficient' values non-negative
  for i, label in enumerate(labels):
    if label in ['steps', 'sleep_coefficient']:
      original_x[:, i] = np.maximum(0, original_x[:, i])

  return original_x


def _get_minute_of_day(time_string: str):
  """Converts a time string in "YYYY-MM-DD HH:MM:SS+ZZ:ZZ" format to the minute of the day (0-1439).

  Args:
    time_string: The time string to convert.

  Returns:
    An integer representing the minute of the day.
    Returns None if the input string is not in the expected format.
  """
  try:
    dt_object = datetime.datetime.fromisoformat(time_string)
    return dt_object.hour * 60 + dt_object.minute
  except ValueError:
    return None


def _get_channel_groups(event) -> Dict[str, Any]:
  """Returns the channel groups for the given event."""
  if event == 'sleep':
    return {
        'Heart': {
            'members': [
                'HR',
                'rr_med',
            ],
            'names': [
                'heart rate',
                'hrv rr',
            ],
        },
        'Sleep': {
            'members': ['sleep_coefficient'],
            'names': ['sleep coefficient'],
        },
    }
  elif event == 'stress':
    return {
        'Heart': {
            'members': [
                'HR',
                'rr_med',
            ],
            'names': [
                'heart rate',
                'hrv rr',
            ],
        },
        'EDA': {
            'members': [
                'eda_level_real',
            ],
            'names': [
                'eda level',
            ],
        },
    }
  elif event == 'Walk':
    return {
        'Heart': {
            'members': [
                'HR',
                'rr_med',
            ],
            'names': [
                'heart rate',
                'hrv rr',
            ],
        },
        'Activity': {
            'members': [
                'steps',
                'log_energy',
            ],
            'names': [
                'steps',
                'log energy',
            ],
        },
    }
  else:
    return {
        'Heart': {
            'members': [
                'HR',
                'rr_med',
            ],
            'names': [
                'heart rate',
                'hrv rr',
            ],
        },
        'Activity': {
            'members': [
                'steps',
                'jerk_auto',
                'log_energy',
            ],
            'names': [
                'steps',
                'jerk',
                'log energy',
            ],
        },
    }


def _average_downsample(arr, target_cols) -> np.ndarray:
  """Downsamples a 2D numpy array using simple averaging.

  Args:
      arr: The input numpy array (2D).
      target_cols: The desired number of columns in the downsampled array.

  Returns:
      The downsampled numpy array.
  """
  rows, cols = arr.shape
  if cols % target_cols != 0:
    raise ValueError('Number of columns must be divisible by target_cols')

  factor = cols // target_cols
  downsampled_arr = arr.reshape(rows, target_cols, factor).mean(axis=2)
  return downsampled_arr


def _update_minute_index(df, n_min=20) -> pd.DataFrame:
  """Updates the index of a DataFrame representing minute-based time-series data.

  Args:
    df: A pandas DataFrame with an integer index representing minutes (0-71).
    n_min: The number of minutes to increment the index by.

  Returns:
    A pandas DataFrame with an updated index representing minutes in increments
    of 20,
    and the index column named "index of minute".
  """

  # Create the new index values by multiplying the existing index by 20.
  new_index = (df.index + 1) * n_min

  # Assign the new index to the DataFrame.
  df.index = new_index

  # Rename the index column.
  df.index.name = 'index of minute'

  return df


def _identify_trends_with_regression(data, downsample_scale=40, top_n=3):
  """Identifies trends by fitting linear regression over segments."""
  trends = []
  thresholds = {6: 1.5, 8: 1.3, 12: 1}

  max_value, min_value = max(data), min(data)
  min_max_diff = max_value - min_value

  for k, v in thresholds.items():
    thresholds[k] = v * min_max_diff / 40

  stable_threshold = 0.01 * min_max_diff

  for segment_size in [6, 8, 12]:
    for i in range(
        0, len(data) - segment_size + 1, segment_size // 2
    ):  # Overlapping segments
      segment = data[i : i + segment_size]
      indices = np.arange(len(segment))
      slope, _, _, _, _ = scipy_stats.linregress(indices, segment)
      start_index = i + 1
      end_index = i + segment_size

      if (
          slope > thresholds[segment_size]
          and data[i + segment_size - 1] - data[i] > 0.2 * min_max_diff
      ):  # Threshold for increasing trend
        trends.append((
            start_index * downsample_scale,
            end_index * downsample_scale,
            'increasing',
            slope,
            data[i + segment_size - 1] - data[i],
            segment_size,
        ))
      elif (
          slope < -thresholds[segment_size]
          and data[i] - data[i + segment_size - 1] > 0.2 * min_max_diff
      ):  # Threshold for decreasing trend
        trends.append((
            start_index * downsample_scale,
            end_index * downsample_scale,
            'decreasing',
            slope,
            data[i] - data[i + segment_size - 1],
            segment_size,
        ))
      elif (
          abs(slope) < stable_threshold
          and max(segment) - min(segment) < 0.1 * min_max_diff
      ):  # Threshold for flat trend
        trends.append((
            start_index * downsample_scale,
            end_index * downsample_scale,
            'stable',
            slope,
            data[i + segment_size - 1],
            segment_size,
        ))

  sorted_trends = sorted(trends, key=lambda item: abs(item[-1]), reverse=True)
  sorted_trends = sorted(
      sorted_trends,
      key=lambda item: abs(item[-3] if item[-4] != 'stable' else 1),
      reverse=True,
  )
  sorted_trends = sorted(
      sorted_trends,
      key=lambda item: abs(item[-2])
      if item[-4] != 'stable'
      else 0.2 * min_max_diff,
      reverse=True,
  )

  selected_trends = []
  for trend in sorted_trends:
    overlap = False
    if len(selected_trends) == top_n:
      break
    start1, end1, _, _, _, _ = trend

    # for each selected trend, check overlap < 0.3
    for start2, end2, _, _, _, _ in selected_trends:
      if max(0, min(end1, end2) - max(start1, start2)) > 0.3 * min(
          end1 - start1, end2 - start2
      ):
        overlap = True
        break
    if not overlap:
      selected_trends.append(trend)

  return selected_trends


def _detect_significant_peaks_valleys(
    data,
    prominence_threshold: float = 0.5,
    height_threshold: float = 0.6,
    distance: int = 5,
    width=(None, 4),
    downsample_scale: int = 40,
) -> list[Any]:
  """Detects significant peaks and valleys in a time series.

  Args:
      data: The time series data.
      prominence_threshold (float, optional): Minimum prominence of
        peaks/valleys. Prominence measures how much a peak stands out due to its
        intrinsic height and its location relative to the nearby lower contour.
        Defaults to None.
      height_threshold (float, optional): Minimum height of peaks (absolute
        value for valleys after inverting). Defaults to None.
      distance (int): Minimum horizontal distance (in data points) between
        neighboring peaks. Smaller peaks within this range are discarded in
        favor of the higher peak. Defaults to 1.
      width: tuple of two ints, optional: Minimum and maximum width in samples
        for peaks.
      downsample_scale: int, optional: the downsample scale of the data.

  Returns:
      tuple: A tuple containing two lists:
              - significant_peaks_indices: Indices of significant peaks.
              - significant_valleys_indices: Indices of significant valleys.
  """

  data = np.array(data)

  results = []

  max_value, min_value, mean_value = max(data), min(data), np.mean(data)
  min_max_diff = max_value - min_value

  lowerbound_threshold = (
      (-mean_value - (1 - height_threshold) * min_max_diff)
      if height_threshold is not None
      else None
  )
  height_threshold = (
      height_threshold * min_max_diff + mean_value
      if height_threshold is not None
      else None
  )

  prominence_threshold = (
      prominence_threshold * min_max_diff
      if prominence_threshold is not None
      else None
  )

  # Find all peaks
  peaks_indices, _ = scipy_signal.find_peaks(
      data,
      prominence=prominence_threshold,
      height=height_threshold,
      distance=distance,
      width=width,
  )
  event_name = random.sample(['peak', 'spike'], 1)[0]
  for peak in peaks_indices:
    results.append(((peak + 1) * downsample_scale, event_name))

  # Find all valleys (by inverting the data)
  inverted_data = -data
  valleys_indices, _ = scipy_signal.find_peaks(
      inverted_data,
      prominence=prominence_threshold,
      height=lowerbound_threshold,
      distance=distance,
      width=width,
  )

  for valley in valleys_indices:
    results.append(((valley + 1) * downsample_scale, 'drop'))

  return results


