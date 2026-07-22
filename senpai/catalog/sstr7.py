"""Low-level readers and spatial queries for the SSTRC7 star catalog."""

import math
import os
import struct
from functools import lru_cache
from typing import Any

import numpy as np
from astropy import wcs

DEFAULT_SSTRC7_PATH = "/data/shared/sstrc7"
RECORD_LEN = 30
RECORD_LEN_BYTES = RECORD_LEN * 2


@lru_cache(maxsize=10)
def load_index(
    filename: str, numRaZones: int = 60, numDecZones: int = 1800
) -> list[list[dict[str, Any]]]:
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
) -> list[dict[str, Any]]:
    """Select a list of regions that intersect with the rectangular bounds.

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

    def append_zones(minRAIndex: int, maxRAIndex: int, bound_func: int) -> None:
        """Append zones spanning the given RA index range to the selection list."""
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
                    selectZone["bound"] = "minRA" if (ra * zoneWidth < ra_max) else "inside"
                elif bound_func == 2:
                    selectZone["bound"] = "maxRA" if (ra * zoneWidth > ra_min) else "inside"

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


def load_zone(currentZone: dict[str, Any], rootPath: str) -> tuple[bytes, int, int]:
    """Load the catalog records for a single index zone.

    Reads the records from the SSTRC catalog data file between the beginning and
    ending records for the specified index entry.

    Args:
        currentZone (dict[str, Any]): index entry with ``id``, ``pos``, and
            ``length`` keys identifying the zone to load.
        rootPath (str): root path to the SSTRC catalog data files.

    Returns:
        tuple[bytes, int, int]: the raw zone buffer, the byte offset of the
            buffer start within the file, and the byte offset of the buffer end.
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
    _id: int,
    pos: int,
    length: int,
    bound: str,
    rootPath: str,
    filter_center: float | None = None,
) -> list[dict[str, Any]]:
    """Load and parse all stars for a specified catalog zone.

    Args:
        _id (int): zone identifier (declination band).
        pos (int): record offset of the zone within its catalog file.
        length (int): number of records in the zone.
        bound (str): bound type for the zone (``"minRA"``, ``"maxRA"``, or
            ``"inside"``); controls iteration order.
        rootPath (str): root path to the SSTRC catalog data files.
        filter_center (float | None): optional wavelength to filter star
            magnitudes by. Defaults to None.

    Returns:
        list[dict[str, Any]]: parsed star records for the zone.
    """
    buffer, start, end = load_zone({"id": _id, "pos": pos, "length": length}, rootPath)
    stars = []
    if bound == "minRA":
        for s in [
            i * RECORD_LEN_BYTES for i in reversed(range((end - start) // RECORD_LEN_BYTES))
        ]:
            star = read_star(buffer[s : s + RECORD_LEN_BYTES], filter_center=filter_center)
            stars.append(star)
    else:
        for s in [i * RECORD_LEN_BYTES for i in range((end - start) // RECORD_LEN_BYTES)]:
            star = read_star(buffer[s : s + RECORD_LEN_BYTES], filter_center=filter_center)
            stars.append(star)

    return stars


def query_by_los(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
    rootPath: str = DEFAULT_SSTRC7_PATH,
    pad_mult: float = 0,
    origin: str = "center",
    filter_ob: bool = True,
    flipud: bool = False,
    fliplr: bool = False,
    filter_center: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query the catalog by focal plane parameters and line-of-sight vector.

    The line of sight vector is defined as the top left corner of the focal
    plane array.

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
        filter_center: `float`, optional wavelength to filter stars by

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
    rootPath: str = DEFAULT_SSTRC7_PATH,
    origin: str = "center",
    filter_center: float | None = None,
) -> list[dict[str, Any]]:
    """Query the catalog by focal plane parameters and line-of-sight vector.

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
    rootPath: str = DEFAULT_SSTRC7_PATH,
    clip_min_max: bool = True,
    filter_center: float | None = None,
) -> list[dict[str, Any]]:
    """Query the catalog by minimum and maximum right ascension and declination.

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        rootPath: `string`, path to root directory. default: environment
            variable SATSIM_SSTR7_PATH
        clip_min_max: `boolean`, clip stars outsize of `ra_min` and `ra_max`
        filter_center: `float`, optional wavelength to filter stars by

    Returns:
        A `list`, stars within the bounds of input parameters
    """
    zoneIndex = load_index(os.path.join(rootPath, "sstrc.acc"), numRaZones=60, numDecZones=1800)

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
        )

        if clip_min_max:

            def clip_stars(
                z: dict[str, Any] = z, ss: list[dict[str, Any]] = ss
            ) -> list[dict[str, Any]]:
                """Trim a zone's stars to those inside the RA bound for that zone."""
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


def read_star(buffer: bytes, filter_center: float | None = None) -> dict[str, Any]:
    """Reads a byte buffer and parses star parameters.

    Args:
        buffer: `list`, byte array of length 60 bytes
        filter_center: `float`, optional wavelength to filter the star
            magnitude by

    Returns:
        A `dict`, the star position and magnitudes
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

    ra, dec, ra_pm, dec_pm, parallax = raw[0:5]

    raw_mv_array = np.asarray(raw_mv) * 1.0e-3

    centers = mv_centers[(raw_mv_array < 32) & (raw_mv_array > -32)]
    mvs = raw_mv_array[(raw_mv_array < 32) & (raw_mv_array > -32)]

    mv = np.interp(filter_center, centers, mvs) if filter_center is not None else get_star_mv(raw_mv_array)

    if mv < -32 or mv > 32:
        mv = 32

    return {
        "ra": ra * angleScale,
        "dec": dec * angleScale,
        "ra_pm": ra_pm * np.cos(dec * angleScale) * properMotionScale,
        "dec_pm": dec_pm * properMotionScale,
        "parallax": parallax * parallaxScale,
        "mv": mv,
        "catalog": ", ".join(catalog_sources) if catalog_sources else "Unknown",
    }


def get_star_mv(mv: list) -> float:
    """Gets the best visual magnitude available to be used for simulation.

    Args:
        mv: `list`, list of star magnitudes, see `read_star`

    Returns:
        A `float`, the visual magnitude
    """
    if mv[0] < 32:  # Open
        return mv[0]
    if mv[5] < 32:  # Johnson_R
        return mv[5]
    if mv[8] < 32:  # Sloan_r
        return mv[8]
    if mv[4] < 32:  # Johnson_V
        return mv[4]
    if mv[3] < 32:  # Johnson_B
        return mv[3]

    return 32


def get_wcs(
    height: int, width: int, y_ifov: float, x_ifov: float, ra: float, dec: float, rot: float = 0.0
) -> wcs.WCS:
    """Get an AstroPy WCS object for RA/Dec to pixel transformations.

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
    offset: list | None = None,
) -> tuple[list[float], list[float], wcs.WCS]:
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
    if offset is None:
        offset = [0, 0]

    crpix = 0

    w = get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot)

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

    [cra, _cdec] = cworld[0]

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


def decode_source_flags(source_flags: int) -> list[str]:
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
    rootPath: str = DEFAULT_SSTRC7_PATH,
    filter_center: float | None = None,
    safety_margin: float = 0.1,  # Add 10% safety margin to ensure complete coverage
) -> list[dict[str, Any]]:
    """Query the catalog by focal plane parameters, line of sight, and rotation.

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

    # Find min/max ra/dec to define bounding box
    min_ra = np.min(ra_corners)
    max_ra = np.max(ra_corners)
    min_dec = np.min(dec_corners)
    max_dec = np.max(dec_corners)

    # Query stars within the bounding box
    cmin = np.radians([min_ra, min_dec])
    cmax = np.radians([max_ra, max_dec])

    # Get stars from catalog
    stars = query_by_min_max(
        cmin[0], cmax[0], cmin[1], cmax[1], rootPath, filter_center=filter_center
    )

    return stars
