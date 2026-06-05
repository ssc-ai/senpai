import numpy as np
from pydantic import BaseModel, model_validator


def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 180) range and handle 180° ambiguity.

    Args:
        angle: Input angle in degrees

    Returns:
        Normalized angle in [0, 180) range
    """
    # First normalize to [0, 180)
    normalized = angle % 180
    return normalized


def angular_difference(angle1: float, angle2: float) -> float:
    """Calculate the minimum angular difference between two streak orientations.

    Since streaks have 180° ambiguity (a line from A to B is the same as B to A),
    we need to consider both orientations when calculating differences.

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        Minimum angular difference in degrees (0 to 90)
    """
    # Normalize both angles to [0, 180)
    a1 = normalize_angle(angle1)
    a2 = normalize_angle(angle2)

    # Calculate all possible differences considering 180° ambiguity
    differences = [
        abs(a1 - a2),  # Direct difference
        abs(a1 - (a2 + 180)),  # a2 rotated by 180°
        abs((a1 + 180) - a2),  # a1 rotated by 180°
        abs(a1 - (a2 - 180)),  # a2 rotated by -180°
        abs((a1 - 180) - a2),  # a1 rotated by -180°
    ]

    # Return the minimum difference
    min_diff = min(differences)

    # Since we're dealing with orientations (not directions),
    # the maximum meaningful difference is 90°
    return min(min_diff, 180 - min_diff)


class StreakMeasurement(BaseModel):
    rotation: float
    length: float
    fwhm: float | None = None

    @model_validator(mode="after")
    def normalize_rotation(self) -> "StreakMeasurement":
        """Ensure rotation is normalized to [0, 180) range."""
        if self.rotation is not None:
            self.rotation = normalize_angle(self.rotation)
        return self


class StreakMeasurements(BaseModel):
    header: StreakMeasurement | None = None
    cross_correlation: StreakMeasurement | None = None
    frame_extraction: StreakMeasurement | None = None
    previous_frame: StreakMeasurement | None = None
    frame_to_frame: StreakMeasurement | None = None
    validation: StreakMeasurement | None = None
    streak_mapping: StreakMeasurement | None = None

    def filtered_results(
        self,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        # Get all non-None values for each attribute
        frames = [
            self.header,
            self.cross_correlation,
            self.streak_mapping,
            self.frame_extraction,
            self.frame_to_frame,
            self.validation,
            self.previous_frame,
        ]

        weights = [0.3, 0.3, 0.8, 0.10, 0.8, 0.8, 0.8]

        rotations = [m.rotation for m in frames if m is not None]
        lengths = [m.length for m in frames if m is not None]
        # fwhm a bit diff because it can be None
        fwhms = [m.fwhm for m in frames if m is not None and m.fwhm is not None]

        # Filter weights to match available measurements
        filtered_weights = [w for m, w in zip(frames, weights, strict=False) if m is not None]

        # Further filter weights for fwhm where the value itself might be None
        fwhm_weights = [w for m, w in zip(frames, weights, strict=False) if m is not None and m.fwhm is not None]

        return rotations, lengths, fwhms, filtered_weights, fwhm_weights

    def mean_measurement(self) -> StreakMeasurement:
        rotations, lengths, fwhms, weights, fwhm_weights = self.filtered_results()

        # Apply weighted average if we have values
        if rotations:
            # For rotations, we need to handle the circular nature of angles
            # Convert angles to vectors, average, then convert back
            angles_rad = np.deg2rad(rotations)
            weights_array = np.array(weights)
            x_mean = np.average(np.cos(2 * angles_rad), weights=weights_array)
            y_mean = np.average(np.sin(2 * angles_rad), weights=weights_array)
            rotation_avg = np.rad2deg(np.arctan2(y_mean, x_mean)) / 2
            # Ensure result is in [0, 180)
            rotation_avg = normalize_angle(rotation_avg)
        else:
            rotation_avg = 0.0

        if lengths:
            length_avg = np.average(lengths, weights=weights)
        else:
            length_avg = 0.0

        if fwhms:
            fwhm_avg = np.average(fwhms, weights=fwhm_weights)
        else:
            fwhm_avg = None

        return StreakMeasurement(
            rotation=rotation_avg,
            length=length_avg,
            fwhm=fwhm_avg,
        )

    def median_measurement(self) -> StreakMeasurement:
        rotations, lengths, fwhms, weights, fwhm_weights = self.filtered_results()

        # For weighted median, we'll use a simple implementation
        def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
            """Calculate the weighted median of a numpy array."""
            if len(values) == 0:
                return 0.0

            # Convert inputs to numpy arrays if they aren't already
            values = np.asarray(values)
            weights = np.asarray(weights)

            # Sort both arrays based on values
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]

            # Calculate cumulative weights
            cum_weights = np.cumsum(sorted_weights)
            # Normalize cumulative weights
            cum_weights = cum_weights / cum_weights[-1]

            # Find the index where cumulative weight exceeds 0.5
            median_index = np.searchsorted(cum_weights, 0.5)
            if median_index >= len(sorted_values):
                median_index = len(sorted_values) - 1

            return sorted_values[median_index]

        # For rotations, we need special handling due to their circular nature
        if rotations:
            # Convert angles to vectors
            angles_rad = np.deg2rad(rotations)
            x_coords = np.cos(2 * angles_rad)
            y_coords = np.sin(2 * angles_rad)

            # Find weighted medians of x and y components
            median_x = weighted_median(x_coords, np.array(weights))
            median_y = weighted_median(y_coords, np.array(weights))

            # Convert back to angle
            rotation_median = np.rad2deg(np.arctan2(median_y, median_x)) / 2
            # Ensure result is in [0, 180)
            rotation_median = normalize_angle(rotation_median)
        else:
            rotation_median = 0.0

        return StreakMeasurement(
            rotation=rotation_median,
            length=(weighted_median(np.array(lengths), np.array(weights)) if lengths else 0.0),
            fwhm=(weighted_median(np.array(fwhms), np.array(fwhm_weights)) if fwhms else None),
        )

    def sigma_clipped_mean_measurement(self, sigma: float = 2.0) -> StreakMeasurement:
        """Calculate mean measurement after removing outliers using sigma clipping.

        Args:
            sigma: Number of standard deviations to use for clipping (default: 2.0)

        Returns:
            StreakMeasurement with sigma-clipped mean values
        """
        rotations, lengths, fwhms, weights, fwhm_weights = self.filtered_results()

        def circular_sigma_clip(angles: np.ndarray, weights: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
            """Apply sigma clipping to circular data (angles)."""
            if len(angles) < 3:
                return angles, weights

            angles = np.asarray(angles)
            weights = np.asarray(weights)

            # First find the largest cluster of similar angles
            # Try each angle as a potential center and count neighbors
            best_count = 0
            best_center = angles[0]
            angle_threshold = 20.0  # Consider angles within 20° as similar

            for center in angles:
                # Calculate angular differences considering 180° ambiguity
                diffs = np.abs(angles - center)
                diffs = np.minimum(diffs, 180 - diffs)

                # Count angles within threshold
                count = np.sum(diffs <= angle_threshold)

                if count > best_count:
                    best_count = count
                    best_center = center

            # Now use the best center for final filtering
            diffs = np.abs(angles - best_center)
            diffs = np.minimum(diffs, 180 - diffs)

            # Use MAD for robust outlier detection
            mad = np.median(diffs)
            threshold = sigma * mad * 1.4826  # Convert MAD to sigma

            mask = diffs <= threshold
            return angles[mask], weights[mask]

        # For rotations, use circular statistics
        if rotations:
            clipped_rotations, clipped_weights = circular_sigma_clip(np.array(rotations), np.array(weights), sigma)
            if len(clipped_rotations) > 0:
                # Convert to radians for vector averaging
                angles_rad = np.deg2rad(clipped_rotations)
                # Calculate weighted mean direction
                x_mean = np.average(np.cos(angles_rad), weights=clipped_weights)
                y_mean = np.average(np.sin(angles_rad), weights=clipped_weights)
                rotation_mean = np.rad2deg(np.arctan2(y_mean, x_mean))
                rotation_mean = normalize_angle(rotation_mean)
            else:
                # If all angles are clipped, use the one with highest weight
                max_weight_idx = np.argmax(weights)
                rotation_mean = rotations[max_weight_idx]
        else:
            rotation_mean = 0.0

        # Apply sigma clipping to lengths using MAD for robustness
        if lengths:
            lengths_arr = np.array(lengths)
            weights_arr = np.array(weights)

            # Calculate weighted median
            med_length = np.median(lengths_arr)

            # Calculate MAD
            deviations = np.abs(lengths_arr - med_length)
            mad = np.median(deviations)

            # MAD to sigma conversion
            mad_to_sigma = 1.4826
            threshold = sigma * mad * mad_to_sigma

            # Find lengths within threshold
            mask = deviations <= threshold
            clipped_lengths = lengths_arr[mask]
            clipped_weights = weights_arr[mask]

            length_mean = (
                np.average(clipped_lengths, weights=clipped_weights) if len(clipped_lengths) > 0 else med_length
            )
        else:
            length_mean = 0.0

        # Apply similar MAD-based clipping to FWHMs
        if fwhms:
            fwhms_arr = np.array(fwhms)
            fwhm_weights_arr = np.array(fwhm_weights)

            med_fwhm = np.median(fwhms_arr)
            deviations = np.abs(fwhms_arr - med_fwhm)
            mad = np.median(deviations)

            threshold = sigma * mad * 1.4826
            mask = deviations <= threshold

            clipped_fwhms = fwhms_arr[mask]
            clipped_fwhm_weights = fwhm_weights_arr[mask]

            fwhm_mean = np.average(clipped_fwhms, weights=clipped_fwhm_weights) if len(clipped_fwhms) > 0 else med_fwhm
        else:
            fwhm_mean = None

        return StreakMeasurement(
            rotation=rotation_mean,
            length=length_mean,
            fwhm=fwhm_mean,
        )
