"""Catalog-related constants and enums."""

from enum import Enum


class CatalogType(Enum):
    """Available catalog types."""

    SSTRC7 = "sstrc7"
    SDSS = "sdss"
    GAIA = "gaia"
    GAIA_LOCAL = "gaia_local"


class SSTRC7Filter(Enum):
    """Available filters in SSTRC7 catalog."""

    GAIA_G = "Gaia_G"
    GAIA_BP = "Gaia_BP"
    GAIA_RP = "Gaia_RP"
    JOHNSON_B = "Johnson_B"
    JOHNSON_V = "Johnson_V"
    JOHNSON_R = "Johnson_R"
    JOHNSON_I = "Johnson_I"
    SLOAN_G = "Sloan_g"
    SLOAN_R = "Sloan_r"
    SLOAN_I = "Sloan_i"
    SLOAN_Z = "Sloan_z"
    TWOMASS_J = "2MASS_J"
    TWOMASS_H = "2MASS_H"
    TWOMASS_KS = "2MASS_Ks"
    WISE_W1 = "WISE_W1"
    WISE_W2 = "WISE_W2"
    WISE_W3 = "WISE_W3"
    WISE_W4 = "WISE_W4"


class SDSSFilter(Enum):
    """Available filters in SDSS catalog."""

    U = "u"
    G = "g"
    R = "r"
    I = "i"  # noqa: E741 - the SDSS i band; the member name is the public API
    Z = "z"


class GaiaFilter(Enum):
    """Available filters in Gaia catalog."""

    G = "G"
    BP = "BP"
    RP = "RP"


def get_filters_for_catalog(catalog_type: CatalogType) -> list[str]:
    """Get available filters for a given catalog type.

    Args:
        catalog_type: The catalog type enum

    Returns:
        List of available filter names for the catalog

    """
    filter_mappings = {
        CatalogType.SSTRC7: SSTRC7Filter,
        CatalogType.SDSS: SDSSFilter,
        CatalogType.GAIA: GaiaFilter,
    }

    if catalog_type in filter_mappings:
        return [f.value for f in filter_mappings[catalog_type]]
    else:
        return []
