"""Internal input-resolution helpers for TEC LOS workflows.

These helpers preserve ``Model3D.compute_los_tec`` behavior while reducing
method complexity in ``model.py``.
"""

from __future__ import annotations

from typing import Callable


def resolve_receiver_positions(
    receiver_positions,
    receiver_format: str | None,
    receiver_csv,
    receiver_listesta,
    receiver_code: str | None,
    default_receiver_format: str | None,
    default_receiver_csv,
    default_receiver_listesta,
    default_receiver_code: str | None,
    load_receiver_positions_csv: Callable,
    load_receiver_positions_listesta: Callable,
):
    """Resolve receiver inputs using explicit args or stored model defaults."""
    if receiver_positions is not None:
        return receiver_positions

    receiver_format = receiver_format or default_receiver_format
    receiver_csv = receiver_csv or default_receiver_csv
    receiver_listesta = receiver_listesta or default_receiver_listesta
    receiver_code = receiver_code or default_receiver_code
    if receiver_format is None:
        receiver_format = "csv" if receiver_csv is not None else "listesta"

    if str(receiver_format).lower() == "csv":
        if receiver_csv is None:
            raise ValueError(
                "receiver_csv must be provided when receiver_format='csv'."
            )
        return load_receiver_positions_csv(
            receiver_csv,
            receiver_code,
        )

    if str(receiver_format).lower() == "listesta":
        if receiver_listesta is None:
            raise ValueError(
                "receiver_listesta must be provided when receiver_format='listesta'."
            )
        return load_receiver_positions_listesta(
            receiver_listesta,
            receiver_code,
        )

    raise ValueError("receiver_format must be one of: 'csv', 'listesta'.")


def resolve_satellite_positions(
    satellite_positions,
    orbit_format: str | None,
    orbit_h5,
    orbit_pos,
    sat_id: str | None,
    constellation: str | None,
    prn: int | None,
    sat_number: int | None,
    satpos_root,
    satpos_date,
    sat_mapping_file,
    start_offset_s: float | None,
    default_orbit_format: str | None,
    default_orbit_h5,
    default_orbit_pos,
    default_sat_id: str | None,
    default_constellation: str | None,
    default_prn: int | None,
    default_sat_number: int | None,
    default_satpos_root,
    default_satpos_date,
    default_sat_mapping_file,
    default_start_offset_s: float | None,
    event_time,
    output_dt_s: float,
    load_orbits_hdf5: Callable,
    build_satpos_file_path: Callable,
    load_orbits_pos: Callable,
):
    """Resolve satellite inputs using explicit args or stored model defaults."""
    if satellite_positions is not None:
        return satellite_positions, sat_id, constellation, prn

    orbit_format = orbit_format or default_orbit_format
    orbit_h5 = orbit_h5 or default_orbit_h5
    orbit_pos = orbit_pos or default_orbit_pos
    sat_id = sat_id or default_sat_id
    constellation = constellation or default_constellation
    prn = prn or default_prn
    sat_number = sat_number or default_sat_number
    satpos_root = satpos_root or default_satpos_root
    satpos_date = satpos_date or default_satpos_date
    sat_mapping_file = sat_mapping_file or default_sat_mapping_file
    start_offset_s = (
        default_start_offset_s
        if start_offset_s is None
        else float(start_offset_s)
    )

    if orbit_format is None:
        orbit_format = "h5" if orbit_h5 is not None else "pos"

    if str(orbit_format).lower() == "h5":
        if orbit_h5 is None:
            raise ValueError(
                "orbit_h5 must be provided when orbit_format='h5'."
            )
        satellite_positions = load_orbits_hdf5(
            orbit_h5,
            event_time=event_time,
            sat_id=sat_id,
            constellation=constellation,
            prn=prn,
            output_dt_s=output_dt_s,
        )
        return satellite_positions, sat_id, constellation, prn

    if str(orbit_format).lower() == "pos":
        if orbit_pos is None:
            if satpos_root is None or satpos_date is None or sat_number is None:
                raise ValueError(
                    "orbit_pos or (satpos_root, satpos_date, sat_number) "
                    "must be provided when orbit_format='pos'."
                )
            orbit_pos = build_satpos_file_path(
                satpos_root,
                str(satpos_date),
                int(sat_number),
            )

        satellite_positions = load_orbits_pos(
            orbit_pos,
            event_time=event_time,
            sat_id=sat_id,
            constellation=constellation,
            prn=prn,
            sat_number=sat_number,
            sat_mapping_file=sat_mapping_file,
            start_offset_s=float(start_offset_s),
            output_dt_s=output_dt_s,
        )
        return satellite_positions, sat_id, constellation, prn

    raise ValueError("orbit_format must be one of: 'h5', 'pos'.")
