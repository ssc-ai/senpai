# Star Catalogs

This module provides access to astronomical star catalogs for photometry and star matching in SENPAI. The catalogs contain precise positions and magnitudes of stars that can be queried based on sky coordinates and magnitude limits.

## Supported Catalogs

### SDSS (Sloan Digital Sky Survey)
- **Query Method**: Online API via `astroquery.sdss`
- **Bands**: u, g, r, i, z
- **Depth**: ~23 magnitudes in g-band
- **Coverage**: Northern sky (δ > -3°)
- **Use Case**: Photometry calibration, star matching in optical images

### Gaia
- **Query Method**: Online API via `astroquery.gaia`
- **Bands**: G, BP, RP
- **Depth**: ~21 magnitudes in G-band
- **Coverage**: Full sky
- **Use Case**: Precise astrometry, proper motion corrections

### SSTRC7 (Space Surveillance Telescope Right Ascension of the Catalog)
- **Query Method**: Local file-based catalog
- **Bands**: Multiple (Gaia G/BP/RP, Johnson B/V/R/I, Sloan g/r/i/z, 2MASS J/H/Ks, WISE W1-W4)
- **Depth**: ~18-20 magnitudes (varies by band)
- **Coverage**: Full sky
- **Use Case**: Deep photometry, multi-band analysis, satellite detection

## Usage

The catalog system is integrated into SENPAI's astrometry and photometry pipelines. Catalogs are queried automatically when performing:

- Astrometric fitting with photometric calibration
- Photometry measurements
- Star field validation
- Satellite streak analysis

### Configuration

Catalog settings are configured in your SENPAI config file:

```yaml
catalog:
  type: "sdss"  # or "gaia" or "sstrc7"
  sstrc7:
    path: "/path/to/sstrc7/catalog"
    filters: ["Gaia_G", "Sloan_g", "Sloan_r"]
  magnitude_limits:
    faint: 20.0
    bright: -5.0
```

### Manual Query Examples

```python
from senpai.catalog import query_catalog

# Query SDSS catalog
stars = query_catalog(
    catalog_type="sdss",
    ra_center=180.0,
    dec_center=0.0,
    radius_deg=1.0,
    faint_lim=20.0,
    bright_lim=10.0
)

# Query Gaia catalog
stars = query_catalog(
    catalog_type="gaia",
    ra_center=180.0,
    dec_center=0.0,
    radius_deg=1.0,
    faint_lim=18.0,
    bright_lim=5.0
)

# Query SSTRC7 catalog (requires local files)
stars = query_catalog(
    catalog_type="sstrc7",
    ra_center=180.0,
    dec_center=0.0,
    radius_deg=1.0,
    faint_lim=18.0,
    bright_lim=5.0,
    catalog_path="/path/to/sstrc7"
)
```

## Catalog Data Structure

All catalog queries return stars in a standardized format:

```python
{
    'ra': float,        # Right ascension in degrees
    'dec': float,       # Declination in degrees
    'mag': float,       # Magnitude in primary band
    'mag_err': float,   # Magnitude error
    'band': str,        # Magnitude band name
    'catalog_id': str,  # Unique identifier in catalog
    'additional_data': dict  # Catalog-specific fields
}
```

## SSTRC7 Local Catalog Setup

For SSTRC7, you need to download and set up the local catalog files:

1. **Download**: Obtain SSTRC7 catalog files from appropriate astronomical data archives
2. **Organization**: Place files in a directory structure expected by the module
3. **Configuration**: Set the `catalog.sstrc7.path` in your config file
4. **Indexing**: The module will automatically index and cache catalog queries for performance

## Performance Considerations

- **SDSS/Gaia**: Network-dependent, cached for repeated queries
- **SSTRC7**: Local file access, fast but requires disk space
- **Caching**: Results are cached to improve performance on repeated queries
- **Memory**: Large sky regions may consume significant memory

## Troubleshooting

### Common Issues

1. **Network timeouts**: For online catalogs, ensure stable internet connection
2. **Missing SSTRC7 files**: Verify catalog path and file integrity
3. **Magnitude limits**: Check that your limits are appropriate for the catalog depth
4. **Coordinate ranges**: Ensure RA is in [0, 360) and DEC is in [-90, 90]

### Catalog Selection Guidelines

- **Precision astrometry**: Use Gaia
- **Optical photometry**: Use SDSS
- **Deep/IR photometry**: Use SSTRC7
- **Offline operation**: Use SSTRC7
- **Quick validation**: Use SDSS
