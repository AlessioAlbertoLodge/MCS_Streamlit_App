"""
A tiny stand-in for the real ENTSO-E Transparency Platform API.
For a handful of hard-coded examples it returns exactly the
XML you would receive from `documentType=A44` and converts it
to a tidy pandas.DataFrame.

Only Germany (DE) and France (FR) on 2025-05-24 are included.
"""

from __future__ import annotations
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

_NS = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

# --------------------------------------------------------------------------- #
# 1) “Fake” API responses – verbatim XML strings (shortened for space)        #
# --------------------------------------------------------------------------- #
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_XML_DE_20250524 = (DATA_DIR / "entsoe_surrogate.de.xml").read_text()
_XML_FR_20250524 = (DATA_DIR / "entsoe_surrogate.fr.xml").read_text()


def _xml_for(country: str, date: dt.date) -> str:
    if country.upper() in ("DE", "GERMANY") and date == dt.date(2025, 5, 24):
        return _XML_DE_20250524
    if country.upper() in ("FR", "FRANCE") and date == dt.date(2025, 5, 24):
        return _XML_FR_20250524
    raise ValueError("No surrogate XML available for that country/date combo.")


# --------------------------------------------------------------------------- #
# 2) XML → pandas                                                             #
# --------------------------------------------------------------------------- #
def parse_entsoe_price_xml(xml_text: str, country: str | None = None) -> pd.DataFrame:
    """
    Parse an ENTSO-E `Publication_MarketDocument` (documentType A44) into
    a DataFrame with timestamped prices.

    Returns columns
    ---------------
    time   : datetime64[ns] – start of each period (UTC)
    price  : float          – €/MWh
    country: string
    """

    root = ET.fromstring(xml_text)

    period_node = root.find(".//ns:Period", _NS)
    start_str = period_node.find("./ns:timeInterval/ns:start", _NS).text
    resolution = period_node.find("./ns:resolution", _NS).text  # e.g. PT60M
    points = period_node.findall("./ns:Point", _NS)

    start_dt = dt.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    step_minutes = int(resolution[2:-1])  # 'PT60M' -> 60

    times: list[dt.datetime] = [
        start_dt + dt.timedelta(minutes=step_minutes) * (i - 1)
        for i in range(1, len(points) + 1)
    ]
    prices = [float(p.find("./ns:price.amount", _NS).text) for p in points]

    if country is None:
        # try to infer from out_Domain, fall back to '??'
        country = root.find(".//ns:out_Domain.mRID", _NS).text or "??"

    return (
        pd.DataFrame({"time": times, "price": prices})
        .assign(country=country)
        .sort_values("time")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# 3) Public helper: get_day_ahead_prices                                      #
# --------------------------------------------------------------------------- #
def get_day_ahead_prices(country: str, date: dt.date) -> pd.DataFrame:
    """
    Fetch a surrogate day-ahead price series *as if* it came from ENTSO-E.

    Parameters
    ----------
    country : ISO-2 code or name (“DE”, “Germany”, “FR”, “France”)
    date    : Python date (YYYY-MM-DD)

    Returns
    -------
    pandas.DataFrame
    """
    xml_text = _xml_for(country, date)
    return parse_entsoe_price_xml(xml_text, country=country)
