import argparse
import math
import os
import struct
from functools import lru_cache

import numpy as np
from astropy import wcs

DEFAULT_SSTR7_PATH = "/path/to/sstrc7"
RECORD_LEN = 30
RECORD_LEN_BYTES = RECORD_LEN * 2


@lru_cache(maxsize=10)
def load_index(filename: str, numRaZones: int = 60, numDecZones: int = 1800):
    """Map SSTRC index file entries to index vector and RA and Dec index maps.
    The SSTRC index files differ from the SSTRC catalog accelerator (index)
    files in that the only the zone position and length are stored as binary
    unsigned integers in a single file.

    Args:
        filename: `int`, index filename.
        numRaZones: `int`, number of RA zones. default: 60
        numDecZones: `int`, number of Dec zones. default 1800

    Returns:
        A `list`, list of dictionaries with zone position and length
    """

    zoneIndex = [[{}] * numRaZones for i in range(numDecZones)]

    with open(filename, mode="rb") as f:
        data = np.fromfile(f, dtype=np.dtype("<u4"))
        i = 0
        for decIndex in range(numDecZones):
            for raIndex in range(numRaZones):
                zoneIndex[decIndex][raIndex] = {"pos": data[i], "length": data[i + 1]}
                i = i + 2

    return zoneIndex


def select_zone(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    zoneIndex: list,
    numRaZones: int = 60,
    numDecZones: int = 1800,
):
    """Select a list of regions that intersect with the rectangular coordinate
    bounds

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        zoneIndex: `list`, list of zones
        numRaZones: `int`, number of RA zones. default: 60
        numDecZones: `int`, number of Dec zones. default 1800

    Returns:
        A `list`, list of dictionaries with zone position and length that
            encompass the ra/dec min max
    """

    decZoneLimit = numDecZones - 1
    raZoneLimit = numRaZones - 1

    zoneHeight = math.pi / numDecZones
    zoneWidth = 2.0 * math.pi / numRaZones

    selectZoneList = []

    spd_min = math.pi / 2.0 + dec_min
    spd_max = math.pi / 2.0 + dec_max

    minSPDIndex = max(int(spd_min / zoneHeight), 0)
    maxSPDIndex = min(int(spd_max / zoneHeight), decZoneLimit)

    minRAIndex = 0
    maxRAIndex = 0

    def append_zones(minRAIndex, maxRAIndex, bound_func):
        for spd in range(minSPDIndex, maxSPDIndex + 1):
            for ra in range(minRAIndex, maxRAIndex + 1):
                selectZone = {
                    "id": int(spd),
                    "pos": zoneIndex[spd][ra]["pos"],
                    "length": zoneIndex[spd][ra]["length"],
                }
                if bound_func == 0:
                    selectZone["bound"] = (
                        "maxRA"
                        if ((ra + 1) * zoneWidth > ra_max)
                        else ("minRA" if (ra * zoneWidth < ra_min) else "inside")
                    )
                elif bound_func == 1:
                    selectZone["bound"] = (
                        "minRA" if (ra * zoneWidth < ra_max) else "inside"
                    )
                elif bound_func == 2:
                    selectZone["bound"] = (
                        "maxRA" if (ra * zoneWidth > ra_min) else "inside"
                    )

                if selectZone["length"] > 0:
                    selectZoneList.append(selectZone)

    # Check to see if catalog search bounds crosses 0 deg Right Ascension and
    # if so, query from the minimum RA coordinate to the end of the SPD band
    # (360 deg RA) and from the start of the SPD band (0 deg RA) to the maximum
    # RA coordinate
    if ra_min <= ra_max:
        minRAIndex = max(int(ra_min / zoneWidth), 0)
        maxRAIndex = min(int(ra_max / zoneWidth), raZoneLimit)
        append_zones(minRAIndex, maxRAIndex, 0)

    # Search bounds cross 0 deg RA
    else:
        minRAIndex = max(int(ra_min / zoneWidth) - 1, 0)
        maxRAIndex = raZoneLimit
        append_zones(minRAIndex, maxRAIndex, 1)

        minRAIndex = 0
        maxRAIndex = min(int(ra_max / zoneWidth) + 1, raZoneLimit)
        append_zones(minRAIndex, maxRAIndex, 2)

    return selectZoneList


def load_zone(currentZone, rootPath):
    """Load the region with the records from the SSTRC catalog data file
    between the beginning and ending records for the specified index entry.
    """
    # Next, assure the proper zone catalog file is opened. Seek to the proper
    # file offset and read the entire zone region into the zone buffer
    filename = os.path.join(rootPath, "s{:04d}.cat".format(currentZone["id"]))

    with open(filename, "rb") as file:
        zoneBufferFilePos = currentZone["pos"]
        zoneBufferOffset = zoneBufferFilePos * RECORD_LEN_BYTES
        file.seek(zoneBufferOffset)

        # Read the region star data into memory
        zoneBufferLen = currentZone["length"] * RECORD_LEN_BYTES
        zoneBuffer = file.read(zoneBufferLen)

        zoneBufferPos = zoneBufferOffset
        zoneBufferEnd = zoneBufferOffset + zoneBufferLen

        return zoneBuffer, zoneBufferPos, zoneBufferEnd


@lru_cache(maxsize=1024)
def load_stars_for_zone(
    id: int,
    pos: int,
    length: int,
    bound: str,
    rootPath: str,
    filter_center: float = None,
    faint_lim: float = None,
    bright_lim: float = None,
):
    """Load stars for specified zone."""
    buffer, start, end = load_zone({"id": id, "pos": pos, "length": length}, rootPath)
    stars = []
    if bound == "minRA":
        for s in [
            i * RECORD_LEN_BYTES
            for i in reversed(range((end - start) // RECORD_LEN_BYTES))
        ]:
            # Quick magnitude check first - skip expensive parsing if star doesn't meet criteria
            if not quick_magnitude_check(
                buffer[s : s + RECORD_LEN_BYTES], faint_lim, bright_lim
            ):
                continue

            star = read_star(
                buffer[s : s + RECORD_LEN_BYTES],
                filter_center=filter_center,
                faint_lim=None,  # Already filtered above
                bright_lim=None,  # Already filtered above
            )
            stars.append(star)
            # if star['ra'] < ra_min:
            #     print('load minRA:', len(stars))
            #     break
    else:
        for s in [
            i * RECORD_LEN_BYTES for i in range((end - start) // RECORD_LEN_BYTES)
        ]:
            # Quick magnitude check first - skip expensive parsing if star doesn't meet criteria
            if not quick_magnitude_check(
                buffer[s : s + RECORD_LEN_BYTES], faint_lim, bright_lim
            ):
                continue

            star = read_star(
                buffer[s : s + RECORD_LEN_BYTES],
                filter_center=filter_center,
                faint_lim=None,  # Already filtered above
                bright_lim=None,  # Already filtered above
            )
            stars.append(star)
            # if bound == 'maxRA' and star['ra'] > ra_max:
            #     print('load maxRA:', len(stars))
            #     break

    return stars


def query_by_los(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
    rootPath: str = DEFAULT_SSTR7_PATH,
    pad_mult: float = 0,
    origin: str = "center",
    filter_ob: bool = True,
    flipud: bool = False,
    fliplr: bool = False,
    filter_center: float = None,
):
    """Query the catalog based on focal plane parameters and ra and dec line
    of sight vector. Line of sight vector is defined as the top left corner
    of the focal plane array.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        ra: `float`, right ascension of top left corner of array, [0,0]
        dec: `float`, declination of top left corner of array, [0,0]
        rot: `float`, focal plane rotation (not implemented)
        rootPath: path to root directory. default: environment variable SATSIM_SSTR7_PATH
        pad_mult: `float`, padding multiplier
        origin: `string`, if `center`, rr and cc will be defined where the line of sight is at the center of
            the focal plane array. default='center'
        filter_ob: `boolean`, remove stars outside pad
        flipud: `boolean`, flip row coordinates
        fliplr: `boolean`, flip column coordinates

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row pixel locations
            cc: `list`, list of column pixel locations
            mv: `list`, list of visual magnitudes
    """

    cmin, cmax, w = get_min_max_ra_dec(
        height, width, y_fov / height, x_fov / width, ra, dec, rot, pad_mult, origin
    )

    cmin = np.radians(cmin)
    cmax = np.radians(cmax)
    stars = query_by_min_max(
        cmin[0], cmax[0], cmin[1], cmax[1], rootPath, filter_center=filter_center
    )

    rra = np.array([s["ra"] for s in stars])
    ddec = np.array([s["dec"] for s in stars])
    mm = np.array([s["mv"] for s in stars])

    cc, rr = w.wcs_world2pix(np.degrees(rra), np.degrees(ddec), 0)

    if filter_ob:
        hp = height * (1 + pad_mult)
        wp = width * (1 + pad_mult)
        in_bounds = np.logical_and.reduce([rr <= hp, rr >= -hp, cc <= wp, cc >= -wp])
        rr = rr[in_bounds]
        cc = cc[in_bounds]
        mm = mm[in_bounds]

    if origin == "center":
        rr += height / 2.0
        cc += width / 2.0

    if flipud:
        rr = height - rr

    if fliplr:
        cc = width - cc

    return rr, cc, mm


def query_by_los_radec(
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rootPath: str = DEFAULT_SSTR7_PATH,
    origin: str = "center",
    filter_center: float = None,
):
    """Query the catalog based on focal plane parameters and ra and dec line
    of sight vector.

    Args:
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        ra: `float`, right ascension of the field center
        dec: `float`, declination of the field center
        rootPath: path to root directory. default: environment variable SATSIM_SSTR7_PATH
        origin: `string`, if `center`, ra and dec represent the center of the field,
            if `corner`, ra and dec represent the top-left corner. default='center'
        filter_center: `float`, optional wavelength to filter stars by

    Returns:
        A `list`, stars within the bounds of input parameters
    """
    if origin == "corner":
        # If ra/dec represent the top-left corner, adjust to get the center
        cmin = [ra, dec]
        cmax = [ra + x_fov, dec + y_fov]
        # Calculate center from corners
        ra = (cmin[0] + cmax[0]) / 2
        dec = (cmin[1] + cmax[1]) / 2

    # Now ra/dec represent the center of the field
    cmin = [ra - x_fov / 2, dec - y_fov / 2]
    cmax = [ra + x_fov / 2, dec + y_fov / 2]

    cmin = np.radians(cmin)
    cmax = np.radians(cmax)
    stars = query_by_min_max(
        cmin[0], cmax[0], cmin[1], cmax[1], rootPath, filter_center=filter_center
    )

    return stars


def query_by_min_max(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    rootPath: str = DEFAULT_SSTR7_PATH,
    clip_min_max: bool = True,
    filter_center: float = None,
    faint_lim: float = None,
    bright_lim: float = None,
):
    """Query the catalog based on focal plane parameters and minimum and
    maximum right ascension and declination.

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        rootPath: `string`, path to root directory. default: environment
            variable SATSIM_SSTR7_PATH
        clip_min_max: `boolean`, clip stars outsize of `ra_min` and `ra_max`

    Returns:
        A `list`, stars within the bounds of input parameters
    """

    zoneIndex = load_index(
        os.path.join(rootPath, "sstrc.acc"), numRaZones=60, numDecZones=1800
    )

    zones = select_zone(
        ra_min, ra_max, dec_min, dec_max, zoneIndex, numRaZones=60, numDecZones=1800
    )

    stars = []
    for z in zones:
        ss = load_stars_for_zone(
            z["id"],
            z["pos"],
            z["length"],
            z["bound"],
            rootPath,
            filter_center=filter_center,
            faint_lim=faint_lim,
            bright_lim=bright_lim,
        )

        if clip_min_max:

            def clip_stars():
                if z["bound"] == "minRA":
                    for i in range(len(ss)):
                        if ss[i]["ra"] < ra_min:
                            return ss[0:i]
                else:
                    # break for `maxRA` and not `inside`
                    for i in range(len(ss)):
                        if z["bound"] == "maxRA" and ss[i]["ra"] > ra_max:
                            return ss[0:i]
                return ss

            ss = clip_stars()

        stars += ss

    return stars


def quick_magnitude_check(
    buffer: bytes, faint_lim: float | None = None, bright_lim: float | None = None
) -> bool:
    """Quickly check if a star passes magnitude limits without full parsing.

    Args:
        buffer: `bytes`, 60-byte star record
        faint_lim: `float`, faint magnitude limit
        bright_lim: `float`, bright magnitude limit

    Returns:
        `bool`, True if star passes magnitude limits, False otherwise
    """
    if faint_lim is None and bright_lim is None:
        return True  # No filtering needed

    # Read just the magnitude data (bytes 14-50, 18 magnitudes)
    raw_mv = struct.unpack("=hhhhhhhhhhhhhhhhhh", buffer[14 : 14 + 18 * 2])
    raw_mv_array = np.asarray(raw_mv) * 1.0e-3

    # Get the best magnitude using the same logic as read_star
    mv = get_star_mv(raw_mv_array)

    # Apply magnitude limits
    if faint_lim is not None and mv >= faint_lim:
        return False
    if bright_lim is not None and mv <= bright_lim:
        return False

    return True


def read_star(
    buffer: bytes,
    filter_center: float | None = None,
    faint_lim: float | None = None,
    bright_lim: float | None = None,
):
    """Reads a byte buffer and parses star parameters.

    Args:
        buffer: `list`, byte array of length 60 bytes
        filter_center: `float`, optional wavelength to filter stars by
        faint_lim: `float`, optional faint magnitude limit (stars fainter than this will be skipped)
        bright_lim: `float`, optional bright magnitude limit (stars brighter than this will be skipped)

    Returns:
        A `dict`, the star position and magnitudes, or None if star is filtered out
    """

    mas2rad = 4.84813681109535993589914102358e-9
    year2sec = 3.1556952e7

    angleScale = mas2rad  # to radians
    properMotionScale = (1 / 0.32) * mas2rad / year2sec  # to radians / sec
    parallaxScale = (1 / 0.032) * mas2rad

    raw = []
    raw = struct.unpack("=iihhh", buffer[0:14])

    # zach, expand with additional filter magnitudes
    raw_mv = []
    raw_mv = struct.unpack("=hhhhhhhhhhhhhhhhhh", buffer[14 : 14 + 18 * 2])

    # Extract source flags
    source_flags = struct.unpack("=H", buffer[52:54])[0]

    # Decode source flags into catalog names
    catalog_sources = decode_source_flags(source_flags)

    mv_centers = np.asarray(
        [
            600,  # open (gaia G)
            500,  # Gaia BP
            800,  # Gaia RP
            440,  # Johnson_B
            548,  # Johnson_V
            700,  # Johnson R
            900,  # Johnson I
            477,  # sloan g
            622,  # Sloan r
            762,  # sloan i
            913,  # sloan z
            1235,  # 2mass J
            1662,  # 2mass h
            2159,  # 2mass k_s
            3400,  # WISE w1
            4600,  # WISE w2
            12000,  # WISE w3
            22000,  # WISE w4
        ]
    )

    filter_names = [
        "Gaia_G",
        "Gaia_BP",
        "Gaia_RP",
        "Johnson_B",
        "Johnson_V",
        "Johnson_R",
        "Johnson_I",
        "Sloan_g",
        "Sloan_r",
        "Sloan_i",
        "Sloan_z",
        "2MASS_J",
        "2MASS_H",
        "2MASS_Ks",
        "WISE_W1",
        "WISE_W2",
        "WISE_W3",
        "WISE_W4",
    ]

    ra, dec, ra_pm, dec_pm, parallax = raw[0:5]

    raw_mv_array = np.asarray(raw_mv) * 1.0e-3

    # STEP 1: Create magnitudes dictionary with ALL available filter magnitudes
    # This is the source of truth - all valid magnitudes go here first
    magnitudes = {}
    for filter_name, mag_value in zip(filter_names, raw_mv_array, strict=False):
        if -32 < mag_value < 32:  # Valid magnitude range
            magnitudes[filter_name] = float(mag_value)

    # STEP 2: Pick a primary magnitude FROM the magnitudes dict
    # Priority order matches get_star_mv() for consistency
    mv = None
    if filter_center is not None:
        # Interpolate from available magnitudes
        # Need to reconstruct centers and mvs arrays for interpolation
        centers = mv_centers[(raw_mv_array < 32) & (raw_mv_array > -32)]
        mvs = raw_mv_array[(raw_mv_array < 32) & (raw_mv_array > -32)]
        if len(centers) > 0 and len(mvs) > 0:
            mv = np.interp(filter_center, centers, mvs)
            if mv < -32 or mv > 32:
                mv = None
    else:
        # Use priority order to pick primary from magnitudes dict
        # This matches get_star_mv() priority: Johnson_V > Johnson_R > Sloan_r > Gaia_G > Sloan_g > Johnson_B
        if "Johnson_V" in magnitudes:
            mv = magnitudes["Johnson_V"]
        elif "Johnson_R" in magnitudes:
            mv = magnitudes["Johnson_R"]
        elif "Sloan_r" in magnitudes:
            mv = magnitudes["Sloan_r"]
        elif "Gaia_G" in magnitudes:
            mv = magnitudes["Gaia_G"]
        elif "Sloan_g" in magnitudes:
            mv = magnitudes["Sloan_g"]
        elif "Johnson_B" in magnitudes:
            mv = magnitudes["Johnson_B"]
        elif len(magnitudes) > 0:
            # Fallback: use first available magnitude
            mv = next(iter(magnitudes.values()))

    # STEP 3: Ensure magnitudes dict is never empty and primary magnitude is set
    # If no valid magnitudes found, this is an error case, but handle gracefully
    if len(magnitudes) == 0:
        # This should be rare - means all magnitudes are invalid
        # Set a sentinel value and log a warning
        mv = 32.0  # Invalid magnitude sentinel
        magnitudes["Invalid"] = 32.0  # Mark as invalid
    else:
        # Ensure mv is set (should already be set above, but double-check)
        if mv is None:
            mv = next(iter(magnitudes.values()))  # Use first available

    # Clamp mv to valid range
    if mv < -32 or mv > 32:
        mv = 32.0

    # CRITICAL: Ensure magnitudes dict is never None or empty
    # This is the source of truth - if magnitude exists, magnitudes must exist
    assert len(magnitudes) > 0, f"magnitudes dict is empty but mv={mv}"
    assert mv is not None, "mv is None but magnitudes dict exists"

    return {
        "ra": ra * angleScale,
        "dec": dec * angleScale,
        "ra_pm": ra_pm * np.cos(dec * angleScale) * properMotionScale,
        "dec_pm": dec_pm * properMotionScale,
        "parallax": parallax * parallaxScale,
        "mv": mv,
        "magnitudes": magnitudes,  # Always a non-empty dict
        "catalog": ", ".join(catalog_sources) if catalog_sources else "Unknown",
    }


def get_star_mv(mv: list) -> float:
    """Gets the best magnitude available for open band (silicon) observations.

    For open band observations, we prioritize magnitudes that best match
    silicon response, which is roughly similar to V-band or R-band.

    Args:
        mv: `list`, list of star magnitudes, see `read_star`

    Returns:
        A `float`, the best available magnitude
    """
    # Priority order for open band (silicon response):
    # 1. Johnson_V (closest to silicon peak response)
    # 2. Johnson_R (good for silicon)
    # 3. Sloan_r (similar to Johnson R)
    # 4. Gaia_G (broad band, but may need color correction)
    # 5. Sloan_g (blueward of silicon peak)
    # 6. Johnson_B (too blue for silicon)

    if mv[4] < 32:  # Johnson_V (index 4)
        return mv[4]
    if mv[5] < 32:  # Johnson_R (index 5)
        return mv[5]
    if mv[8] < 32:  # Sloan_r (index 8)
        return mv[8]
    if mv[0] < 32:  # Gaia_G (index 0) - broad band
        return mv[0]
    if mv[7] < 32:  # Sloan_g (index 7)
        return mv[7]
    if mv[3] < 32:  # Johnson_B (index 3)
        return mv[3]

    return 32


def get_wcs(
    height: int,
    width: int,
    y_ifov: float,
    x_ifov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
):
    """Get an AstroPy world coordinate system (WCS) object used to transform
    RA, Dec coordinates to focal plane array pixel coordinates.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_ifov: `float`, y i-fov in degrees
        x_ifov: `float`, x i-fov in degrees
        ra: `float`, right ascension at pixel 0,0
        dec: `float`, declination of at pixel 0,0
        rot: `float`, rotation of the focal plane in degrees

    Returns:
        A `WCS`, used to transform RA, Dec to pixel coordinates
    """

    # TODO move to center
    crpix = [1, 1]

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = crpix
    w.wcs.cdelt = np.array([x_ifov, y_ifov])
    w.wcs.crval = np.array([ra, dec])
    w.wcs.crota = [rot, rot]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return w


def get_min_max_ra_dec(
    height: int,
    width: int,
    y_ifov: float,
    x_ifov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
    pad_mult: float = 0.0,
    origin: str = "center",
    offset: list = [0, 0],
):
    """Get the min and max RA and Dec bounds based on focal plane parameters.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_ifov: `float`, y ifov in degrees
        x_ifov: `float`, x ifov in degrees
        ra: `float`, right ascension of `origin`
        dec: `float`, declination of `origin`
        rot: `float`, focal plane rotation
        pad_mult: `float`, padding multiplier
        origin: `string`, corner or center
        offset: `float`, array specifying [row, col] offset in pixels

    Returns:
        A `tuple`, containing:
            cmin: `array`, minimum ra and dec
            cmax: `array`, maximum ra and dec
            wcs: `WCS`, used to transform RA, Dec to pixel coordinates

    """

    crpix = 0

    w = get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot)

    # pixcrd = np.array([[1,1],[1,height+1],[width+1,1],[width+1,height+1]], np.float_)
    hp = height * pad_mult
    wp = width * pad_mult
    pixcrd = np.array(
        [
            [crpix - wp, crpix - hp],
            [crpix - wp, crpix + height * 0.5],
            [crpix - wp, crpix + height + hp],
            [crpix + width * 0.5, crpix + height + hp],
            [crpix + width + wp, crpix + height + hp],
            [crpix + width + wp, crpix + height * 0.5],
            [crpix + width + wp, crpix - hp],
            [crpix + width * 0.5, crpix - hp],
        ],
        np.float64,
    )

    center = np.array([[width / 2.0, height / 2.0]])

    if origin == "center":
        pixcrd[:, 0] -= width / 2.0 + offset[1]
        pixcrd[:, 1] -= height / 2.0 + offset[0]
        center[:, 0] -= width / 2.0 + offset[1]
        center[:, 1] -= height / 2.0 + offset[0]

    # Convert pixel coordinates to world coordinates
    world = w.wcs_pix2world(pixcrd, 1)
    cworld = w.wcs_pix2world(center, 1)

    [cra, cdec] = cworld[0]

    [minTheta, minPhi] = np.min(world, axis=0)
    [maxTheta, maxPhi] = np.max(world, axis=0)

    northpole = w.wcs_world2pix([[0, 89.99999]], 1)[0]
    southpole = w.wcs_world2pix([[0, -89.99999]], 1)[0]

    if (
        not np.any(np.isnan(northpole))
        and northpole[0] > 0
        and northpole[0] < width
        and northpole[1] > 0
        and northpole[1] < height
    ):
        cmin = [0, minPhi]
        cmax = [360.0, 90.0]
    elif (
        not np.any(np.isnan(southpole))
        and southpole[0] > 0
        and southpole[0] < width
        and southpole[1] > 0
        and southpole[1] < height
    ):
        cmin = [0, -90.0]
        cmax = [360.0, maxPhi]
    elif cra > maxTheta or cra < minTheta or (maxTheta - minTheta) > 180:
        # Including theta meridian crossing
        cmin = [maxTheta, minPhi]
        cmax = [minTheta, maxPhi]
    else:
        cmin = [minTheta, minPhi]
        cmax = [maxTheta, maxPhi]

    return cmin, cmax, w


def decode_source_flags(source_flags):
    """Decode source flags into a list of catalog and property names.

    Args:
        source_flags: Integer containing the bit flags

    Returns:
        List of strings describing the catalogs and properties
    """
    flag_meanings = {
        0x0001: "Bright Star Catalog (HR)",
        0x0002: "Henry Draper Catalog (HD)",
        0x0004: "Hipparcos Catalog",
        0x0008: "Tycho-Gaia (TGAS) Catalog",
        0x0010: "Gaia Catalog",
        0x0020: "Landolt Catalog",
        0x0040: "2MASS Catalog",
        0x0080: "AllWISE Catalog",
        0x0100: "Astrometric Standard",
        0x0200: "Extended Source",
        0x0400: "High Proper Motion Star",
        0x0800: "Multiple stars",
        0x1000: "Photometric Standard",
        0x2000: "Spectrophotometric Star",
        0x4000: "Variable star",
        0x8000: "SWIR Standard",
    }

    active_flags = []
    for flag_value, flag_name in flag_meanings.items():
        if source_flags & flag_value:
            active_flags.append(flag_name)

    return active_flags


def query_by_los_radec_with_rotation(
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rotation: float = 0.0,
    rootPath: str = DEFAULT_SSTR7_PATH,
    filter_center: float = None,
    faint_lim: float = None,
    bright_lim: float = None,
    safety_margin: float = 0.1,  # Add 10% safety margin to ensure complete coverage
):
    """Query the catalog based on focal plane parameters, ra and dec line
    of sight vector, and field rotation.

    Args:
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        ra: `float`, right ascension of the field center in degrees
        dec: `float`, declination of the field center in degrees
        rotation: `float`, field rotation in degrees
        rootPath: path to root directory
        filter_center: `float`, optional wavelength to filter stars by
        safety_margin: `float`, fraction to expand the search area by

    Returns:
        A `list`, stars within the bounds of input parameters with full star data
    """
    # Apply safety margin to FOV
    x_fov_with_margin = x_fov * (1 + safety_margin)
    y_fov_with_margin = y_fov * (1 + safety_margin)

    # Calculate the corners of the field in ra/dec space
    half_width = x_fov_with_margin / 2
    half_height = y_fov_with_margin / 2

    # Define corners in pixel space (relative to center)
    corners_rel = np.array(
        [
            [-half_width, -half_height],  # bottom-left
            [half_width, -half_height],  # bottom-right
            [half_width, half_height],  # top-right
            [-half_width, half_height],  # top-left
        ]
    )

    # Rotation matrix (counter-clockwise)
    rot_rad = np.radians(rotation)
    rot_matrix = np.array(
        [[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]]
    )

    # Apply rotation to corners
    rotated_corners = np.dot(corners_rel, rot_matrix.T)

    # Add center coordinates to get absolute positions
    ra_corners = ra + rotated_corners[:, 0] / np.cos(
        np.radians(dec)
    )  # Correct for spherical distortion
    dec_corners = dec + rotated_corners[:, 1]

    # Find min/max dec
    min_dec = np.min(dec_corners)
    max_dec = np.max(dec_corners)

    # Normalize RA corners to [0, 360) range
    ra_corners_normalized = np.mod(ra_corners, 360.0)

    # Check if field crosses RA = 0/360 boundary
    # This happens when the span of normalized corners is much larger than the FOV
    ra_span = np.max(ra_corners_normalized) - np.min(ra_corners_normalized)
    crosses_zero = ra_span > 180.0  # If span > 180°, we must have crossed the boundary

    if crosses_zero:
        # Field crosses RA = 0/360 boundary
        # Find the RA values on each side of the boundary
        ra_low_side = ra_corners_normalized[
            ra_corners_normalized < 180.0
        ]  # Values near 0
        ra_high_side = ra_corners_normalized[
            ra_corners_normalized >= 180.0
        ]  # Values near 360

        if len(ra_low_side) > 0 and len(ra_high_side) > 0:
            # We need to query two ranges: [min_high, 360) and [0, max_low)
            min_ra_high = np.min(ra_high_side)
            max_ra_low = np.max(ra_low_side)

            # Query first range: high RA values (near 360)
            cmin1 = np.radians([min_ra_high, min_dec])
            cmax1 = np.radians([360.0, max_dec])
            stars1 = query_by_min_max(
                cmin1[0],
                cmax1[0],
                cmin1[1],
                cmax1[1],
                rootPath,
                filter_center=filter_center,
                faint_lim=faint_lim,
                bright_lim=bright_lim,
            )

            # Query second range: low RA values (near 0)
            cmin2 = np.radians([0.0, min_dec])
            cmax2 = np.radians([max_ra_low, max_dec])
            stars2 = query_by_min_max(
                cmin2[0],
                cmax2[0],
                cmin2[1],
                cmax2[1],
                rootPath,
                filter_center=filter_center,
                faint_lim=faint_lim,
                bright_lim=bright_lim,
            )

            # Combine results
            stars = stars1 + stars2
        else:
            # Shouldn't happen, but fall back to simple query
            min_ra = np.min(ra_corners_normalized)
            max_ra = np.max(ra_corners_normalized)
            cmin = np.radians([min_ra, min_dec])
            cmax = np.radians([max_ra, max_dec])
            stars = query_by_min_max(
                cmin[0],
                cmax[0],
                cmin[1],
                cmax[1],
                rootPath,
                filter_center=filter_center,
                faint_lim=faint_lim,
                bright_lim=bright_lim,
            )
    else:
        # Field doesn't cross boundary - simple case
        min_ra = np.min(ra_corners_normalized)
        max_ra = np.max(ra_corners_normalized)

        # Query stars within the bounding box
        cmin = np.radians([min_ra, min_dec])
        cmax = np.radians([max_ra, max_dec])
        stars = query_by_min_max(
            cmin[0],
            cmax[0],
            cmin[1],
            cmax[1],
            rootPath,
            filter_center=filter_center,
            faint_lim=faint_lim,
            bright_lim=bright_lim,
        )

    return stars


def filter_catalog_by_magnitude(
    source_path: str,
    dest_path: str,
    faint_lim: float,
    bright_lim: float = -32.0,
    numRaZones: int = 60,
    numDecZones: int = 1800,
    verbose: bool = True,
):
    """Filter an SSTRC7 catalog by magnitude and save to a new location.

    This utility reads an existing SSTRC7 catalog, filters stars by magnitude,
    and writes a new filtered catalog with updated index files. Preserves the
    RA/Dec zone structure for efficient spatial queries.

    Args:
        source_path: `str`, path to source catalog directory
        dest_path: `str`, path to destination catalog directory (will be created)
        faint_lim: `float`, faint magnitude limit (stars fainter than this are excluded)
        bright_lim: `float`, bright magnitude limit (stars brighter than this are excluded).
            Default: -32.0 (includes all bright stars)
        numRaZones: `int`, number of RA zones. default: 60
        numDecZones: `int`, number of Dec zones. default: 1800
        verbose: `bool`, print progress information

    Returns:
        None. Creates filtered catalog files in dest_path.

    Example:
        >>> filter_catalog_by_magnitude(
        ...     source_path="/path/to/sstrc7",
        ...     dest_path="/path/to/sstrc7_mag15",
        ...     faint_lim=15.0
        ... )
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)

    if verbose:
        print(f"Filtering catalog from {source_path} to {dest_path}")
        print(f"Magnitude range: {bright_lim} < mag <= {faint_lim}")

    # Load the original index to preserve RA zone structure
    source_index_file = os.path.join(source_path, "sstrc.acc")
    source_zone_index = load_index(
        source_index_file, numRaZones=numRaZones, numDecZones=numDecZones
    )

    # Initialize new zone index
    new_zone_index = []

    total_kept = 0
    total_original = 0

    # Process each Dec zone
    for dec_zone_id in range(numDecZones):
        source_file = os.path.join(source_path, f"s{dec_zone_id:04d}.cat")
        dest_file = os.path.join(dest_path, f"s{dec_zone_id:04d}.cat")

        if not os.path.exists(source_file):
            if verbose and dec_zone_id % 100 == 0:
                print(f"Warning: Zone file {source_file} does not exist, skipping")
            # Add empty zones to index
            for _ in range(numRaZones):
                new_zone_index.extend([0, 0])  # pos=0, length=0
            continue

        # Process each RA zone within this Dec zone
        filtered_records = []
        ra_zone_info = []  # Store (start_pos, length) for each RA zone

        for ra_zone_id in range(numRaZones):
            zone_info = source_zone_index[dec_zone_id][ra_zone_id]
            zone_pos = zone_info["pos"]
            zone_length = zone_info["length"]

            if zone_length == 0:
                ra_zone_info.append((len(filtered_records), 0))
                continue

            # Read this specific RA zone's data
            with open(source_file, "rb") as f:
                f.seek(zone_pos * RECORD_LEN_BYTES)
                zone_data = f.read(zone_length * RECORD_LEN_BYTES)

            # Filter stars in this RA zone
            ra_zone_filtered = []
            for i in range(zone_length):
                start_idx = i * RECORD_LEN_BYTES
                end_idx = start_idx + RECORD_LEN_BYTES
                record = zone_data[start_idx:end_idx]

                # Check magnitude
                if quick_magnitude_check(
                    record, faint_lim=faint_lim, bright_lim=bright_lim
                ):
                    ra_zone_filtered.append(record)

            # Record where this RA zone starts in the filtered output and its length
            ra_zone_start = len(filtered_records)
            ra_zone_len = len(ra_zone_filtered)
            ra_zone_info.append((ra_zone_start, ra_zone_len))

            # Add filtered records to the output list
            filtered_records.extend(ra_zone_filtered)

            total_original += zone_length
            total_kept += ra_zone_len

        # Write filtered zone file
        if filtered_records:
            with open(dest_file, "wb") as f:
                for record in filtered_records:
                    f.write(record)

        # Add RA zone info to index
        for ra_start, ra_len in ra_zone_info:
            new_zone_index.extend([ra_start, ra_len])

        if verbose and dec_zone_id % 100 == 0:
            zone_kept = sum(info[1] for info in ra_zone_info)
            zone_original = sum(
                source_zone_index[dec_zone_id][ra]["length"] for ra in range(numRaZones)
            )
            print(
                f"Processed Dec zone {dec_zone_id}/{numDecZones}: {zone_kept}/{zone_original} stars kept"
            )

    # Write new index file
    index_file = os.path.join(dest_path, "sstrc.acc")
    with open(index_file, "wb") as f:
        # Convert to numpy array and write as unsigned 32-bit integers
        index_array = np.array(new_zone_index, dtype=np.uint32)
        index_array.tofile(f)

    if verbose:
        print("\nFiltering complete!")
        print(
            f"Total stars: {total_kept:,}/{total_original:,} kept ({100 * total_kept / max(total_original, 1):.1f}%)"
        )
        print(f"New catalog written to: {dest_path}")
        print(f"Index file: {index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter SSTRC7 catalog by magnitude and save to a new directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter catalog to only include stars with magnitude <= 15
  python sstr7.py /path/to/sstrc7 /path/to/sstrc7_mag15 --faint-lim 15.0

  # Filter catalog with both bright and faint limits
  python sstr7.py /path/to/sstrc7 /path/to/sstrc7_mag10to15 --faint-lim 15.0 --bright-lim 10.0

  # Quiet mode (no progress output)
  python sstr7.py /path/to/sstrc7 /path/to/sstrc7_mag15 --faint-lim 15.0 --quiet
        """,
    )

    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to source SSTRC7 catalog directory",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory for filtered catalog (will be created if doesn't exist)",
    )

    parser.add_argument(
        "--faint-lim",
        type=float,
        required=True,
        help="Faint magnitude limit (stars fainter than this will be excluded)",
    )

    parser.add_argument(
        "--bright-lim",
        type=float,
        default=-32.0,
        help="Bright magnitude limit (stars brighter than this will be excluded). Default: -32.0 (no bright limit)",
    )

    parser.add_argument(
        "--num-ra-zones",
        type=int,
        default=60,
        help="Number of RA zones in catalog. Default: 60",
    )

    parser.add_argument(
        "--num-dec-zones",
        type=int,
        default=1800,
        help="Number of declination zones in catalog. Default: 1800",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate that source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist: {args.source_dir}")
        exit(1)

    # Check for index file in source directory
    index_file = os.path.join(args.source_dir, "sstrc.acc")
    if not os.path.exists(index_file):
        print(f"Error: Index file not found: {index_file}")
        print("Source directory may not be a valid SSTRC7 catalog")
        exit(1)

    # Run the filter
    filter_catalog_by_magnitude(
        source_path=args.source_dir,
        dest_path=args.output_dir,
        faint_lim=args.faint_lim,
        bright_lim=args.bright_lim,
        numRaZones=args.num_ra_zones,
        numDecZones=args.num_dec_zones,
        verbose=not args.quiet,
    )
