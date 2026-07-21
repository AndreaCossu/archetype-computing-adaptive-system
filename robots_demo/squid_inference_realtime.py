import argparse
import os
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib
import numpy as np

from squid_inference import animate_prediction_trajectory
from train_squid_center_cycle import (
    load_center_cycle_for_inference,
    predict_center_cycle,
)


TIME_COLUMN_NAMES = {'time', 't', 'timestamp'}
IGNORED_TARGET_COLUMN_NAMES = {'x', 'y', 'theta'}


def file_signature(path):
    """Return the modification-time and size signature for a file.

    :param path: File path to inspect.
    :return: Tuple ``(mtime_ns, size)``.
    """
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def wait_until_file_settles(path, settle_seconds):
    """Wait until a file signature remains unchanged across one interval.

    :param path: File path to monitor.
    :param settle_seconds: Seconds to wait between signature checks.
    :return: Stable file signature.
    """
    previous_signature = file_signature(path)
    while True:
        time.sleep(settle_seconds)
        current_signature = file_signature(path)
        if current_signature == previous_signature:
            return current_signature
        previous_signature = current_signature


def split_column_line(line, delimiter):
    """Split one header or data line into stripped columns.

    :param line: Input text line.
    :param delimiter: Optional delimiter; whitespace splitting is used when
        ``None``.
    :return: List of stripped column strings.
    """
    if delimiter is None:
        return line.strip().split()
    return [column.strip() for column in line.strip().split(delimiter)]


def is_numeric_row(columns):
    """Return whether a row contains at least one numeric value.

    Empty columns are ignored.

    :param columns: Iterable of column strings.
    :return: ``True`` when every non-empty column parses as a float.
    """
    saw_value = False
    try:
        for column in columns:
            if column == '':
                continue
            float(column)
            saw_value = True
    except ValueError:
        return False
    return saw_value


def normalize_column_name(column_name):
    """Normalize a column name for layout detection.

    :param column_name: Raw column name.
    :return: Lowercase stripped column name.
    """
    return column_name.strip().lower()


def read_text_time_series_file(path, delimiter, skip_header):
    """Read a delimited text time-series file.

    The first non-numeric line is treated as a header and skipped
    automatically.

    :param path: Text file path.
    :param delimiter: Optional delimiter passed to NumPy.
    :param skip_header: Minimum number of header lines to skip.
    :return: Tuple ``(data, column_names, first_data_file_row)``.
    """
    column_names = None
    auto_skip_header = 0
    with open(path, 'r') as input_file:
        first_line = input_file.readline()

    if first_line:
        first_columns = split_column_line(first_line, delimiter)
        if not is_numeric_row(first_columns):
            column_names = first_columns
            auto_skip_header = 1

    effective_skip_header = max(skip_header, auto_skip_header)
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=effective_skip_header,
        dtype=np.float32,
        filling_values=np.nan,
    )
    return data, column_names, effective_skip_header + 1


def read_time_series_file(path, delimiter, skip_header, npz_key):
    """Read a time-series table from text, ``.npy``, or ``.npz`` input.

    :param path: Input file path.
    :param delimiter: Optional delimiter for text input.
    :param skip_header: Header rows skipped for text input.
    :param npz_key: Optional key for ``.npz`` archives.
    :return: Tuple ``(data, column_names, first_data_file_row)``.
    :raises ValueError: If the loaded data is not two-dimensional.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == '.npy':
        data = np.load(path)
        column_names = None
        first_data_file_row = 1
    elif extension == '.npz':
        archive = np.load(path)
        if npz_key is None:
            npz_key = archive.files[0]
        data = archive[npz_key]
        column_names = None
        first_data_file_row = 1
    else:
        data, column_names, first_data_file_row = read_text_time_series_file(
            path,
            delimiter=delimiter,
            skip_header=skip_header,
        )

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(
            f'Expected a 2D time-series table, got shape {data.shape}'
        )
    if column_names is not None and len(column_names) != data.shape[1]:
        column_names = None
    return data, column_names, first_data_file_row


def checkpoint_feature_indices(inference_state):
    """Extract ordered input feature indices from a checkpoint state.

    :param inference_state: State returned by
        ``load_center_cycle_for_inference``.
    :return: Ordered unique feature indices expected by the model.
    :raises ValueError: If no feature indices are present.
    """
    feature_indices = []
    for stats in inference_state['normalization_stats']['features']:
        for feature_idx in stats['feature_idxs']:
            if feature_idx not in feature_indices:
                feature_indices.append(feature_idx)
    if not feature_indices:
        raise ValueError('Checkpoint does not define input feature indices')
    return feature_indices


def make_expected_time_series(feature_values, time_values, feature_indices):
    """Build the model input table expected by center-cycle inference.

    :param feature_values: Compact feature matrix ordered like
        ``feature_indices``.
    :param time_values: Time column values.
    :param feature_indices: Model feature positions in the checkpoint layout.
    :return: Time column plus zero-filled model feature columns.
    :raises ValueError: If the compact feature count does not match.
    """
    expected_no_time_columns = max(feature_indices) + 1
    if feature_values.shape[1] != len(feature_indices):
        raise ValueError(
            f'Expected {len(feature_indices)} input feature columns, '
            f'got {feature_values.shape[1]}'
        )

    model_without_time = np.zeros(
        (feature_values.shape[0], expected_no_time_columns),
        dtype=np.float32,
    )
    model_without_time[:, feature_indices] = feature_values
    return np.column_stack(
        [
            np.asarray(time_values, dtype=np.float32),
            model_without_time,
        ]
    ).astype(np.float32, copy=False)


def generated_time_values(row_count):
    """Generate default monotonic time values for rows without timestamps.

    :param row_count: Number of rows.
    :return: Float32 array ``[0, 1, ..., row_count - 1]``.
    """
    return np.arange(row_count, dtype=np.float32)


def finite_true_positions_or_none(true_positions):
    """Return finite true positions or ``None`` when unavailable.

    :param true_positions: Optional true ``x,y`` position array.
    :return: Float32 positions, or ``None`` if missing/all-NaN.
    """
    if true_positions is None:
        return None
    true_positions = np.asarray(true_positions, dtype=np.float32)
    if not np.isfinite(true_positions).any():
        return None
    return true_positions


def adapt_named_time_series(data, column_names, feature_indices):
    """Adapt a named-column table to the checkpoint input layout.

    :param data: Raw numeric table.
    :param column_names: Column names from the input file.
    :param feature_indices: Model feature positions in the checkpoint layout.
    :return: Tuple ``(model_time_series, true_positions, layout_description)``;
        ``model_time_series`` is ``None`` when the named layout is incompatible.
    """
    normalized_names = [
        normalize_column_name(column_name)
        for column_name in column_names
    ]
    time_column = next(
        (
            column_idx
            for column_idx, column_name in enumerate(normalized_names)
            if column_name in TIME_COLUMN_NAMES
        ),
        None,
    )

    true_columns = [
        next(
            (
                column_idx
                for column_idx, column_name in enumerate(normalized_names)
                if column_name == target_name
            ),
            None,
        )
        for target_name in ('x', 'y')
    ]
    true_positions = None
    if all(column_idx is not None for column_idx in true_columns):
        true_positions = finite_true_positions_or_none(data[:, true_columns])

    ignored_columns = set()
    if time_column is not None:
        ignored_columns.add(time_column)
    for column_idx, column_name in enumerate(normalized_names):
        if column_name in IGNORED_TARGET_COLUMN_NAMES:
            ignored_columns.add(column_idx)

    input_feature_columns = [
        column_idx
        for column_idx in range(data.shape[1])
        if column_idx not in ignored_columns
    ]
    if len(input_feature_columns) != len(feature_indices):
        return None, true_positions, None

    if time_column is None:
        time_values = generated_time_values(len(data))
        layout_description = (
            'Using named sensor columns with generated time values'
        )
    else:
        time_values = data[:, time_column]
        if 'theta' in normalized_names:
            layout_description = (
                'Using named sensor columns; x/y/theta are ignored for '
                'model input'
            )
        elif true_positions is None:
            layout_description = (
                'Using named time and sensor columns'
            )
        else:
            layout_description = (
                'Using named time,x,y and sensor columns'
            )

    model_time_series = make_expected_time_series(
        data[:, input_feature_columns],
        time_values,
        feature_indices,
    )
    return model_time_series, true_positions, layout_description


def adapt_numeric_time_series(data, feature_indices):
    """Adapt a numeric table by inferring one of the supported layouts.

    Supported layouts include full tables with time/x/y(/theta), compact
    feature-only tables, and compact tables with a leading time column.

    :param data: Raw numeric table.
    :param feature_indices: Model feature positions in the checkpoint layout.
    :return: Tuple ``(model_time_series, true_positions, layout_description)``.
    :raises ValueError: If the column count does not match any supported
        layout.
    """
    expected_no_time_columns = max(feature_indices) + 1
    expected_time_columns = expected_no_time_columns + 1
    full_without_theta_time_columns = expected_time_columns - 1
    full_without_theta_no_time_columns = expected_no_time_columns - 1
    compact_feature_columns = len(feature_indices)
    column_count = data.shape[1]

    if column_count == expected_time_columns:
        return (
            make_expected_time_series(
                data[:, 4:],
                data[:, 0],
                feature_indices,
            ),
            finite_true_positions_or_none(data[:, 1:3]),
            'Using full numeric layout with time,x,y,theta columns',
        )
    if column_count == full_without_theta_time_columns:
        return (
            make_expected_time_series(
                data[:, 3:],
                data[:, 0],
                feature_indices,
            ),
            finite_true_positions_or_none(data[:, 1:3]),
            'Using full numeric layout with time,x,y and sensor columns',
        )
    if column_count == full_without_theta_no_time_columns:
        return (
            make_expected_time_series(
                data[:, 2:],
                generated_time_values(len(data)),
                feature_indices,
            ),
            finite_true_positions_or_none(data[:, 0:2]),
            'Using full numeric layout with x,y and sensor columns',
        )
    if column_count == compact_feature_columns + 1:
        return (
            make_expected_time_series(
                data[:, 1:],
                data[:, 0],
                feature_indices,
            ),
            None,
            'Using compact numeric layout with time and sensor columns only',
        )
    if column_count == compact_feature_columns:
        return (
            make_expected_time_series(
                data,
                generated_time_values(len(data)),
                feature_indices,
            ),
            None,
            'Using compact numeric layout with sensor columns only',
        )

    raise ValueError(
        f'Input has {column_count} columns, but this checkpoint expects '
        f'either {full_without_theta_time_columns} full columns '
        f'(time,x,y plus sensors), {full_without_theta_no_time_columns} '
        f'full columns without time, {compact_feature_columns + 1} compact '
        f'columns with time, or {compact_feature_columns} compact sensor '
        f'columns. Legacy {expected_time_columns}-column input with theta '
        f'is also accepted, but theta is set to 0.'
    )


def adapt_time_series_for_inference(data, column_names, inference_state):
    """Adapt raw input data to the model layout used for inference.

    Named-column detection is attempted first when names are available, then
    numeric-layout inference is used as a fallback.

    :param data: Raw numeric table.
    :param column_names: Optional input column names.
    :param inference_state: Loaded checkpoint inference state.
    :return: Tuple ``(model_time_series, true_positions, layout_description)``.
    """
    feature_indices = checkpoint_feature_indices(inference_state)
    if column_names is not None:
        model_time_series, true_positions, layout_description = (
            adapt_named_time_series(data, column_names, feature_indices)
        )
        if model_time_series is not None:
            return model_time_series, true_positions, layout_description

    return adapt_numeric_time_series(data, feature_indices)


def predict_window_targeting_row(time_series, target_row_index, inference_state, window_size):
    """Predict the target position for a row using prior-window context.

    :param time_series: Adapted model input table.
    :param target_row_index: Row index whose prediction should be emitted.
    :param inference_state: Loaded checkpoint inference state.
    :param window_size: Number of prior rows required for the prediction.
    :return: Predicted ``x,y`` values, or ``None`` when insufficient history is
        available.
    """
    window_end = target_row_index
    if window_end < window_size:
        return None

    latest_window = time_series[window_end - window_size:window_end]
    module_predictions = predict_center_cycle(
        latest_window[None, :, :],
        inference_state,
        cycle_passes=1,
        average_modules=False,
    )
    return module_predictions[0, -1]


def data_row_to_file_row(path, data_row_index, first_data_file_row):
    """Map a zero-based data row index to a user-facing file row number.

    :param path: Input file path.
    :param data_row_index: Zero-based row index in the loaded data array.
    :param first_data_file_row: One-based first data row for text files.
    :return: One-based row number for logging.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension in {'.npy', '.npz'}:
        return data_row_index + 1
    return first_data_file_row + data_row_index


def save_realtime_gif(
    predictions,
    gif_file,
    interval_seconds,
    true_positions=None,
):
    """Write an animated trajectory GIF atomically.

    :param predictions: Sequence of predicted ``x,y`` positions.
    :param gif_file: Output GIF path.
    :param interval_seconds: Frame interval for the animation.
    :param true_positions: Optional true ``x,y`` positions to overlay.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    update_numbers = np.arange(len(predictions))
    if true_positions is not None:
        true_positions = np.asarray(true_positions, dtype=np.float32)
        if not np.isfinite(true_positions).any():
            true_positions = None
    gif_dir = os.path.dirname(os.path.abspath(gif_file))
    os.makedirs(gif_dir, exist_ok=True)
    temporary_gif = os.path.join(
        gif_dir,
        f'.{os.path.basename(gif_file)}.tmp.gif',
    )
    animate_prediction_trajectory(
        predictions,
        update_numbers,
        gif_file=temporary_gif,
        interval_seconds=interval_seconds,
        true_positions=true_positions,
        show_full_history=True,
    )
    os.replace(temporary_gif, gif_file)


def normalize_target_values(values, normalization_stats):
    """Normalize target-space values with checkpoint statistics.

    :param values: Values in original target units.
    :param normalization_stats: Checkpoint normalization metadata.
    :return: Normalized values.
    """
    target_stats = normalization_stats['target']
    mean = np.asarray(target_stats['mean'], dtype=np.float32)
    std = np.asarray(target_stats['std'], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    values = np.asarray(values, dtype=np.float32)
    return (values - mean) / std


def save_normalized_realtime_gif(
    predictions,
    gif_file,
    interval_seconds,
    normalization_stats,
    true_positions=None,
):
    """Normalize predictions and save a trajectory GIF.

    :param predictions: Sequence of predicted positions in original units.
    :param gif_file: Output GIF path.
    :param interval_seconds: Frame interval for the animation.
    :param normalization_stats: Checkpoint normalization metadata.
    :param true_positions: Optional true positions in original units.
    """
    normalized_predictions = normalize_target_values(
        predictions,
        normalization_stats,
    )
    normalized_true_positions = None
    if true_positions is not None:
        normalized_true_positions = normalize_target_values(
            true_positions,
            normalization_stats,
        )
    save_realtime_gif(
        normalized_predictions,
        gif_file=gif_file,
        interval_seconds=interval_seconds,
        true_positions=normalized_true_positions,
    )


def save_realtime_gifs(
    predictions,
    args,
    normalization_stats,
    true_positions=None,
):
    """Save raw and optionally normalized realtime trajectory GIFs.

    :param predictions: Sequence of predicted positions.
    :param args: Parsed CLI namespace with GIF output settings.
    :param normalization_stats: Checkpoint normalization metadata.
    :param true_positions: Optional true positions.
    """
    save_realtime_gif(
        predictions,
        gif_file=args.gif_file,
        interval_seconds=args.animation_interval,
        true_positions=true_positions,
    )
    if not args.normalized_gif:
        return

    save_normalized_realtime_gif(
        predictions,
        gif_file=args.normalized_gif_file,
        interval_seconds=args.animation_interval,
        normalization_stats=normalization_stats,
        true_positions=true_positions,
    )


def gif_true_positions_or_none(predictions, true_positions):
    """Return true positions only when aligned with predictions.

    :param predictions: Prediction history.
    :param true_positions: True-position history.
    :return: Float32 true positions, or ``None`` if unusable.
    """
    if len(true_positions) != len(predictions):
        return None
    true_positions = np.asarray(true_positions, dtype=np.float32)
    if not np.isfinite(true_positions).any():
        return None
    return true_positions


def normalized_target_rmse(prediction, true_position, normalization_stats):
    """Compute RMSE in normalized target space.

    :param prediction: Predicted target values in original units.
    :param true_position: True target values in original units.
    :param normalization_stats: Checkpoint normalization metadata.
    :return: Normalized root mean squared error.
    """
    normalized_prediction = normalize_target_values(
        prediction,
        normalization_stats,
    )
    normalized_true_position = normalize_target_values(
        true_position,
        normalization_stats,
    )
    return np.sqrt(
        np.mean(np.square(normalized_prediction - normalized_true_position))
    )


def default_normalized_gif_file(gif_file):
    """Create a default normalized-GIF path from the raw GIF path.

    :param gif_file: Raw GIF output path.
    :return: Path with ``_normalized`` before the extension.
    """
    gif_root, gif_extension = os.path.splitext(gif_file)
    if gif_extension:
        return f'{gif_root}_normalized{gif_extension}'
    return f'{gif_file}_normalized.gif'


def gif_update_message(args, prediction_count):
    """Build the inline log suffix for a GIF update.

    :param args: Parsed CLI namespace with GIF output settings.
    :param prediction_count: Number of predictions written to the GIF.
    :return: Message suffix.
    """
    if args.normalized_gif:
        return (
            f'; updated {args.gif_file} and {args.normalized_gif_file} '
            f'with {prediction_count} predictions'
        )
    return f'; updated {args.gif_file} with {prediction_count} predictions'


def final_gif_update_message(args, prediction_count):
    """Build the final log message for a last GIF update.

    :param args: Parsed CLI namespace with GIF output settings.
    :param prediction_count: Number of predictions written to the GIF.
    :return: Final update message.
    """
    if args.normalized_gif:
        return (
            f'Updated {args.gif_file} and {args.normalized_gif_file} with '
            f'{prediction_count} predictions before stopping'
        )
    return (
        f'Updated {args.gif_file} with '
        f'{prediction_count} predictions before stopping'
    )


def should_save_gif_for_batch(
    batch_row_count,
    batch_processed_rows,
    batch_prediction_count,
    save_interval,
):
    """Return whether a GIF should be refreshed for a batch position.

    :param batch_row_count: Number of newly available rows in the batch.
    :param batch_processed_rows: Number of rows processed in the batch.
    :param batch_prediction_count: Number of predictions emitted in the batch.
    :param save_interval: Refresh interval for large batches.
    :return: ``True`` when the GIF should be saved now.
    """
    if batch_row_count <= save_interval:
        return True
    return (
        batch_prediction_count % save_interval == 0
        or batch_processed_rows == batch_row_count
    )


def run_realtime_inference(args):
    """Watch an input file and emit center-cycle predictions as rows arrive.

    :param args: Parsed CLI namespace containing file, checkpoint, polling,
        window, GIF, and device settings.
    """
    if not args.show:
        matplotlib.use('Agg')

    inference_state = load_center_cycle_for_inference(
        args.checkpoint_file,
        device=args.device,
    )
    predictions = []
    true_positions = []
    last_signature = None
    last_seen_row_count = None
    last_layout_description = None
    prediction_count = 0

    print(f'Watching {args.input_file}')
    while True:
        if not os.path.exists(args.input_file):
            time.sleep(args.poll_interval)
            continue

        current_signature = file_signature(args.input_file)
        if current_signature == last_signature:
            time.sleep(args.poll_interval)
            continue

        try:
            last_signature = wait_until_file_settles(
                args.input_file,
                args.settle_seconds,
            )
            time_series = read_time_series_file(
                args.input_file,
                delimiter=args.delimiter,
                skip_header=args.skip_header,
                npz_key=args.npz_key,
            )
            raw_time_series, column_names, first_data_file_row = time_series
            model_time_series, file_true_positions, layout_description = (
                adapt_time_series_for_inference(
                    raw_time_series,
                    column_names,
                    inference_state,
                )
            )
        except Exception as exc:
            print(f'Could not process {args.input_file}: {exc}')
            time.sleep(args.poll_interval)
            continue

        if layout_description != last_layout_description:
            print(layout_description)
            last_layout_description = layout_description

        current_row_count = len(model_time_series)
        if last_seen_row_count is None:
            last_seen_row_count = current_row_count
            print(
                f'Waiting for new rows; '
                f'found {current_row_count} existing rows'
            )
            time.sleep(args.poll_interval)
            continue

        if current_row_count < last_seen_row_count:
            last_seen_row_count = current_row_count
            print(
                f'Row count decreased to {current_row_count}; '
                f'waiting for new rows'
            )
            time.sleep(args.poll_interval)
            continue

        batch_start_row = last_seen_row_count
        batch_row_count = current_row_count - batch_start_row
        batch_prediction_count = 0

        for data_row_index in range(batch_start_row, current_row_count):
            last_seen_row_count = data_row_index + 1
            batch_processed_rows = data_row_index - batch_start_row + 1
            file_row = data_row_to_file_row(
                args.input_file,
                data_row_index,
                first_data_file_row,
            )
            prediction = predict_window_targeting_row(
                model_time_series,
                data_row_index,
                inference_state,
                args.window_size,
            )
            if prediction is None:
                print(
                    f'Waiting for {args.window_size} prior input rows; '
                    f'found {data_row_index} before row {file_row}'
                )
                continue

            predictions.append(prediction)
            current_true_position = None
            if file_true_positions is not None:
                current_true_position = file_true_positions[data_row_index]
                if not np.isfinite(current_true_position).all():
                    current_true_position = np.array(
                        [np.nan, np.nan],
                        dtype=np.float32,
                    )
                true_positions.append(current_true_position)
            prediction_count += 1
            batch_prediction_count += 1
            gif_updated = should_save_gif_for_batch(
                batch_row_count=batch_row_count,
                batch_processed_rows=batch_processed_rows,
                batch_prediction_count=batch_prediction_count,
                save_interval=args.gif_batch_interval,
            )
            if gif_updated:
                gif_true_positions = gif_true_positions_or_none(
                    predictions,
                    true_positions,
                )
                save_realtime_gifs(
                    predictions,
                    args,
                    inference_state['normalization_stats'],
                    true_positions=gif_true_positions,
                )
            message = (
                f'Prediction {prediction_count}: '
                f'row={file_row}, '
                f'predicted x={prediction[0]:.4f}, '
                f'predicted y={prediction[1]:.4f}'
            )
            if (
                current_true_position is not None
                and np.isfinite(current_true_position).all()
            ):
                message += (
                    f', true x={current_true_position[0]:.4f}, '
                    f'true y={current_true_position[1]:.4f}'
                )
                if args.verbose:
                    rmse = normalized_target_rmse(
                        prediction,
                        current_true_position,
                        inference_state['normalization_stats'],
                    )
                    message += f', rmse={rmse:.4f}'
            if gif_updated:
                message += gif_update_message(args, len(predictions))
            else:
                message += (
                    f'; deferred GIF update '
                    f'({batch_prediction_count} predictions in this batch; '
                    f'saving every {args.gif_batch_interval})'
                )
            print(message)

            if args.max_updates is not None and prediction_count >= args.max_updates:
                if not gif_updated:
                    gif_true_positions = gif_true_positions_or_none(
                        predictions,
                        true_positions,
                    )
                    save_realtime_gifs(
                        predictions,
                        args,
                        inference_state['normalization_stats'],
                        true_positions=gif_true_positions,
                    )
                    print(
                        final_gif_update_message(args, len(predictions))
                    )
                return

        time.sleep(args.poll_interval)


def main():
    """Parse CLI arguments and start realtime squid inference.

    :raises ValueError: If CLI values are inconsistent or out of range.
    """
    parser = argparse.ArgumentParser(
        description='Watch a time-series file and update a squid prediction GIF'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Continuously updated file containing one raw time step per row',
    )
    parser.add_argument(
        '--checkpoint_file',
        type=str,
        default='center_cycle_checkpoint.pt',
        help='Checkpoint produced by train_squid_center_cycle.py',
    )
    parser.add_argument(
        '--gif_file',
        type=str,
        default='squid_realtime_prediction_trajectory.gif',
    )
    parser.add_argument(
        '--normalized_gif_file',
        type=str,
        default=None,
        help=(
            'GIF file for normalized predicted/true center trajectory when '
            '--normalized_gif is active; defaults to --gif_file with '
            '_normalized before the extension'
        ),
    )
    parser.add_argument(
        '--normalized_gif',
        action='store_true',
        help='Also save a normalized predicted/true center trajectory GIF',
    )
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--poll_interval', type=float, default=0.2)
    parser.add_argument('--settle_seconds', type=float, default=0.05)
    parser.add_argument('--animation_interval', type=float, default=0.2)
    parser.add_argument(
        '--gif_batch_interval',
        type=int,
        default=10,
        help=(
            'When more than this many rows arrive at once, update the GIF '
            'only every N predictions and at the end of the batch'
        ),
    )
    parser.add_argument('--delimiter', type=str, default=',')
    parser.add_argument('--skip_header', type=int, default=0)
    parser.add_argument('--npz_key', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help=(
            'Print normalized per-prediction RMSE when true x,y targets '
            'are available'
        ),
    )
    parser.add_argument(
        '--max_updates',
        type=int,
        default=None,
        help='Stop after this many predictions; mainly useful for tests',
    )
    args = parser.parse_args()
    if args.normalized_gif and args.normalized_gif_file is None:
        args.normalized_gif_file = default_normalized_gif_file(args.gif_file)

    if args.normalized_gif and (
        os.path.abspath(args.normalized_gif_file)
        == os.path.abspath(args.gif_file)
    ):
        raise ValueError('--normalized_gif_file must differ from --gif_file')
    if args.window_size < 1:
        raise ValueError('--window_size must be at least 1')
    if args.poll_interval <= 0:
        raise ValueError('--poll_interval must be greater than 0')
    if args.settle_seconds <= 0:
        raise ValueError('--settle_seconds must be greater than 0')
    if args.animation_interval <= 0:
        raise ValueError('--animation_interval must be greater than 0')
    if args.gif_batch_interval < 1:
        raise ValueError('--gif_batch_interval must be at least 1')
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError('--max_updates must be at least 1')

    run_realtime_inference(args)


if __name__ == '__main__':
    main()
