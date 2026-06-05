# Photometry System

## Overview
The photometry system is designed to extract rich photometric information from astronomical images to support:
- Sensor zero point measurements
- Extinction curve measurements  
- Alt/Az observability map construction
- Photometric calibration and standardization

## Architecture

### Core Components

1. **Single Frame Photometry** (`utils.py`) ✅ **IMPLEMENTED**
   - Extract photometric measurements from individual stars
   - Calculate aperture photometry with multiple apertures
   - Measure background and noise properties
   - Estimate photometric uncertainties
   - Handle crowded field photometry

2. **FWHM Integration** (`cli/sidereal.py`) ✅ **IMPLEMENTED**
   - `measure_fwhm_from_catalog_stars()`: Reusable FWHM measurement function
   - Integrates with existing astrometry pipeline
   - Populates `starfield.fwhm_stats` for photometry use

3. **Multi-Frame Analysis** (future)
   - Cross-frame photometric consistency
   - Time series photometry
   - Variability detection

4. **Calibration Pipeline** (future)
   - Zero point determination
   - Color term corrections
   - Extinction measurements

5. **Observability Maps** (future)
   - Alt/Az coverage analysis
   - Sky brightness mapping
   - Atmospheric transparency monitoring

## Data Flow

1. **Input**: FITS image + WCS solution (from astrometry pipeline)
2. **Star Detection**: Use existing point source extraction
3. **FWHM Measurement**: Measure FWHM from well-isolated catalog stars
4. **Photometric Extraction**: Measure fluxes, backgrounds, uncertainties
5. **Quality Assessment**: Flag bad measurements, estimate reliability
6. **Output**: Rich photometric catalog with metadata

## Photometric Measurements

### Aperture Photometry ✅ **IMPLEMENTED**
- Multiple aperture sizes (e.g., 1×FWHM, 2×FWHM, 3×FWHM)
- Optimal aperture selection based on SNR
- Aperture corrections

### Background Estimation ✅ **IMPLEMENTED**
- Local background measurement using annulus
- Sky level determination (median/mean/mode)
- Background gradient correction

### Uncertainty Estimation ✅ **IMPLEMENTED**
- Poisson noise from source
- Read noise contribution
- Background uncertainty
- Aperture uncertainty
- Systematic errors

### Quality Metrics ✅ **IMPLEMENTED**
- Signal-to-noise ratio
- Crowding indicators
- Saturation flags
- Edge proximity warnings

### FWHM Integration ✅ **IMPLEMENTED**
- Automatic FWHM measurement from catalog stars
- Integration with existing astrometry pipeline
- Proper population of `starfield.fwhm_stats`

## Integration Points

- **Input**: Uses `StarField` from astrometry pipeline
- **Star Positions**: Leverages WCS solutions for accurate coordinates
- **FWHM**: Uses `starfield.fwhm_stats.median_fwhm` for aperture sizing
- **Catalog Matching**: Correlates with photometric standard catalogs

## Usage Examples

### Basic Photometry
```python
from senpai.engine.photometry.utils import measure_starfield_photometry, PhotometryConfig
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.starfield import StarField

# Load your image and starfield
image = load_fits_file("image.fits")
starfield = solve_field(sources)  # From astrometry pipeline

# Perform photometry
config = PhotometryConfig()
results, summary = measure_starfield_photometry(image, starfield, config)

# Access results
print(f"Measured {summary.n_stars} stars")
print(f"Quality measurements: {summary.n_quality}")
print(f"Zero point: {summary.zero_point:.3f} ± {summary.zero_point_err:.3f}")
```

### Integration with Existing Pipeline
```python
from senpai.engine.detection.point.fwhm import measure_fwhm_from_catalog_stars
from senpai.engine.photometry.utils import measure_starfield_photometry

# After astrometry and catalog query...
fwhm_stats = measure_fwhm_from_catalog_stars(image, catalog.stars, initial_fwhm, config)
starfield.fwhm_stats = fwhm_stats

# Now perform photometry with proper FWHM information
results, summary = measure_starfield_photometry(image, starfield, config)
```

### CLI Usage
```bash
# Basic photometry
python -m senpai.cli.photometry -f image.fits -o results.json

# With custom configuration
python -m senpai.cli.photometry -f image.fits -o results.json \
    --aperture-radii 1.0 1.5 2.0 3.0 --min-snr 5.0

# With visualization
python -m senpai.cli.photometry -f image.fits -o results.json \
    --verbose --save-plots --save-apertures
```

### Integration Example
```bash
# Process with both astrometry and photometry
python -m senpai.engine.photometry.integration_example image.fits output_dir/
```

### Test the System
```bash
# Run the test script
python -m senpai.engine.photometry.test_photometry

# Test FWHM integration
python -m senpai.engine.photometry.test_integration
```

## Configuration

The `PhotometryConfig` class allows customization of:

- **Aperture Radii**: List of aperture sizes as multiples of FWHM
- **Background Estimation**: Method and annulus parameters
- **Quality Thresholds**: SNR, crowding, saturation limits
- **Uncertainty Estimation**: Read noise, gain, systematic errors

## Output Format

The photometry system produces rich output including:

### Individual Star Results
- Position (x, y, RA, Dec)
- Multi-aperture flux measurements
- Optimal aperture selection
- Background level and uncertainty
- Quality flags and metrics
- Additional measurements (FWHM, ellipticity, sky coverage)

### Summary Statistics
- Total stars measured
- Quality measurement count
- Median SNR and background
- Limiting magnitude estimate
- Photometric zero point and uncertainty
- FWHM statistics

### FWHM Information
- Median FWHM from catalog stars
- Number of FWHM measurements
- FWHM vs position, magnitude, counts
- Oversampling analysis

## Key Improvements

### FWHM Integration ✅
- **Problem Solved**: Previously, `starfield.fwhm_stats` was not populated after `solve_field()`
- **Solution**: Added `measure_fwhm_from_catalog_stars()` function that can be called after astrometry
- **Benefit**: Photometry now has access to accurate FWHM measurements for proper aperture sizing

### Unified Workflow ✅
- **Problem Solved**: FWHM measurement logic was duplicated between sidereal CLI and integration example
- **Solution**: Extracted reusable function in `cli/sidereal.py`
- **Benefit**: Consistent FWHM measurement across all use cases

### Proper Data Flow ✅
- **Problem Solved**: Photometry was looking for FWHM in `detection_metadata.pixel_fwhm`
- **Solution**: Updated to use `starfield.fwhm_stats.median_fwhm`
- **Benefit**: More accurate and comprehensive FWHM information

## Future Extensions

- PSF photometry for crowded fields
- Difference imaging photometry
- Multi-band photometry
- Real-time photometric monitoring
- Extinction curve measurement pipeline
- Alt/Az observability map generation 