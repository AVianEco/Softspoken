"""
review_exporter.py
------------------

Minimal skeleton for exporting the ReviewDetectionsScreen data into one or
many application‑specific formats.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Callable, Union

import pandas as pd
import os

import soundfile as sf
import wave

def _wav_duration(path: Union[str, Path]) -> float:
    with wave.open(str(path), "rb") as wfh:
        return wfh.getnframes() / float(wfh.getframerate())

def _wav_duration(path: Union[str, Path]) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


class Transform(ABC):
    """
    Base class for a *single* application‑specific export.

    Implement `__call__`, which receives a *copy* of the canonical
    DataFrame and should **return** one of:

    * `pandas.DataFrame`  – manager will save it as CSV.
    * `str` / `bytes`     – manager will write it verbatim.
    * `None`              – transform handled its own file IO.

    Adjust `name` and `extension` to taste.
    """

    name: str = "unnamed"
    extension: str = ".csv"

    @abstractmethod
    def __call__(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError


class ReviewExportManager:
    """Holds the canonical review DataFrame and delegates export work."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._registry: Dict[str, Transform] = {}

    # ---- registration -------------------------------------------------
    def register_transform(self, transform: Transform) -> None:
        if transform.name in self._registry:
            raise KeyError(f"Transform '{transform.name}' already registered")
        self._registry[transform.name] = transform

    # Convenience decorator, lets you do:
    #   @exporter.transform
    #   class FancyTransform(Transform): ...
    def transform(self, cls: type[Transform]) -> type[Transform]:
        self.register_transform(cls())
        return cls

    # ---- export helpers ----------------------------------------------
    def export(
        self,
        name: str,
        dst: str | Path,
        make_dirs: bool = True,
        **kwargs,
    ) -> Path:
        """
        Run ONE registered transform and write its output.

        *dst* may be a directory or a filename.  Returns the final Path.
        """
        if name not in self._registry:
            raise KeyError(f"No transform named '{name}' registered")

        transform = self._registry[name]
        dst = Path(dst)

        if dst.is_dir():
            dst = dst / f"review{transform.extension}"

        if make_dirs:
            dst.parent.mkdir(parents=True, exist_ok=True)

        result = transform(self.df.copy(), **kwargs)

        # --- post‑processing ------------------------------------------
        if isinstance(result, pd.DataFrame):
            result.to_csv(dst, index=False)
        elif isinstance(result, (str, bytes)):
            mode = "w" if isinstance(result, str) else "wb"
            with dst.open(mode) as fh:
                fh.write(result)
        elif result is None:
            # Transform handled its own persistence – do nothing.
            pass
        else:
            raise TypeError(
                "Unsupported return type from transform "
                f"({type(result).__name__})."
            )

        return dst

    def export_all(self, dst_dir: str | Path, **kwargs):
        """
        Run *every* registered transform, writing each one into *dst_dir*.
        Returns a dict:  {transform_name: output_path}
        """
        paths = {}
        for name in self._registry:
            paths[name] = self.export(name, dst_dir, **kwargs)
        return paths


class AudacityTxtTransform(Transform):
    """
    One .txt “label” file per .wav file, compatible with
    Audacity’s **File ▸ Import ▸ Labels…** feature.

    ───────────────────────────────
    Directory layout that will be produced
    ───────────────────────────────
        <base_dir>/
            Audacity Outputs/
                <project_name>/
                    *.txt   # one file per .wav
    ───────────────────────────────
    File format
    ───────────────────────────────
        • Tab‑separated
        • No header line
        • Columns:
              start_time    end_time    "Human"
    """

    name = "audacity"       # used when registering / exporting
    extension = ".txt"      # not critical – every .wav gets its own file

    # ------------------------------------------------------------------
    #  Main entry point – required by Transform
    # ------------------------------------------------------------------
    def __call__(
        self,
        df: pd.DataFrame,
        *,
        base_dir: Union[str, Path],
        project_name: str,
        comment: str = "Human",
        precision: int = 6,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        df : pandas.DataFrame
            The canonical review table (a *copy* supplied by
            ReviewExportManager – edit freely).
        base_dir : str | Path              (required, keyword‑only)
            Folder under which “Audacity Outputs/<project_name>/”
            will be created.
        project_name : str                 (required, keyword‑only)
            Used in the output path.
        comment : str, default "Human"
            Third column written to every label row.
        precision : int, default 6
            Number of decimal places for the time stamps.
        """
        base_dir = Path(base_dir)
        out_root = base_dir / "Audacity Outputs" / project_name
        out_root.mkdir(parents=True, exist_ok=True)

        # --- sanity checks -------------------------------------------------
        required_cols = {"file_name", "start_time", "end_time"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"AudacityTxtTransform: DataFrame missing column(s): {missing}"
            )

        # Ensure numeric and sorted
        df = df.copy()
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["end_time"]   = pd.to_numeric(df["end_time"],   errors="coerce")
        df.sort_values(["file_name", "start_time"], inplace=True)

        # --- write one label file per wav ---------------------------------
        fmt = f"{{:.{precision}f}}\t{{:.{precision}f}}\t{comment}"

        for wav_file, group in df.groupby("file_name", sort=False):
            label_path = out_root / f"{Path(wav_file).stem}.txt"

            lines = [
                fmt.format(row.start_time, row.end_time)
                for row in group.itertuples()
            ]

            # Audacity is fine with a trailing newline
            label_path.write_text("\n".join(lines) + "\n")

        # Return *None* → manager knows we handled all file IO ourselves.
        return None


class KaleidoscopeCsvTransform(Transform):
    """
    Single‑file CSV export for Wildlife Acoustics Kaleidoscope (Pro/Lite).

    ───────────────────────────────
    Output location
    ───────────────────────────────
        <base_dir>/
            Kaleidoscope Outputs/
                <project_name>/
                    <project_name>.csv

    ───────────────────────────────
    Mandatory columns produced
    ───────────────────────────────
        INDIR
        FOLDER
        IN FILE*
        OFFSET
        DURATION
        TOP1MATCH*
        MANUAL ID

    ───────────────────────────────
    Extra columns copied across
    ───────────────────────────────
        end_time
        erase
        review_datetime
    """

    name = "kaleidoscope"
    extension = ".csv"

    # ------------------------------------------------------------------
    #  Main entry point – called by ReviewExportManager
    # ------------------------------------------------------------------
    def __call__(
        self,
        df: pd.DataFrame,
        *,
        base_dir: Union[str, Path],
        project_name: str,
        precision: int = 6,
        human_label: str = "Human",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Canonical review table (the Manager passes a *copy*).
        base_dir : str | Path          (required, keyword‑only)
            Folder under which “Kaleidoscope Outputs/<project_name>/”
            will be created.
        project_name : str             (required, keyword‑only)
            Used to name both the sub‑folder and the .csv file.
        precision : int, default 6
            Rounds OFFSET / DURATION / end_time to this many decimals.
        human_label : str, default "Human"
            Value placed in every TOP1MATCH* cell.
        """

        base_dir = Path(base_dir)
        out_root = base_dir / "Kaleidoscope Outputs" / project_name
        out_root.mkdir(parents=True, exist_ok=True)

        # ------------- sanity checks ------------------------------------
        required_cols = {"file_path", "file_name", "start_time", "end_time"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"KaleidoscopeCsvTransform: DataFrame missing column(s): {missing}"
            )

        # Work on a copy to avoid mutating the caller’s DataFrame
        df = df.copy()

        # Ensure proper dtypes
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["end_time"]   = pd.to_numeric(df["end_time"],   errors="coerce")

        # ----------------------------------------------------------------
        #  Compute INDIR  (longest common path prefix, incl. trailing '\')
        # ----------------------------------------------------------------
        all_paths = df["file_path"].astype(str).tolist()
        indir = os.path.commonpath(all_paths)
        if not indir.endswith(os.sep):
            indir += os.sep  # keep the trailing backslash so FOLDER has no leading sep

        # ----------------------------------------------------------------
        #  Assemble Kaleidoscope table
        # ----------------------------------------------------------------
        out = pd.DataFrame({
            "INDIR":       indir,
            "FOLDER":      [os.path.relpath(p, indir) for p in all_paths],
            "IN FILE*":    df["file_name"],
            "OFFSET":      df["start_time"].round(precision),
            "DURATION":    (df["end_time"] - df["start_time"]).round(precision),
            "TOP1MATCH*":  human_label,
            "MANUAL ID":   df.get("user_comment", pd.Series([""] * len(df))),
            # extra columns for traceability
            "end_time":         df["end_time"].round(precision),
            "erase":            df.get("erase", pd.Series([""] * len(df))),
            "review_datetime":  df.get("review_datetime", pd.Series([""] * len(df))),
        })

        # ----------------------------------------------------------------
        #  Write to disk – one CSV per *project*
        # ----------------------------------------------------------------
        out_path = out_root / f"{project_name}.csv"
        out.to_csv(out_path, index=False)

        # Return None → Manager knows we’ve already handled persistence.
        return None


class RavenTxtTransform(Transform):
    """
    Produces the two files Raven Lite/Pro expect:

        <base_dir>/Raven Outputs/<project_name>/
            ├── <project_name>_listfile.txt   (no header, absolute paths)
            └── <project_name>.txt            (tab‑delimited results table)

    Contiguous timing is built by summing the *actual* length (in seconds) of
    every preceding file in the list file.  Durations are measured with
    `soundfile` if installed, otherwise with the Python standard‑library
    `wave` module (WAV only).
    """

    name = "raven"
    extension = ".txt"      # not really used – we write two filenames

    # ------------------------------------------------------------------
    #  Main entry point (called by ReviewExportManager)
    # ------------------------------------------------------------------
    def __call__(  # noqa: C901  (a bit long but still clear)
        self,
        df: pd.DataFrame,
        *,
        base_dir: Union[str, Path],
        project_name: str,
        precision: int = 6,
        annotation_label: str = "Human",
        low_freq: int = 0,
        high_freq: int = 8000,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Softspoken review DataFrame (copy supplied by manager).
        base_dir : str | Path          (required, keyword‑only)
            Destination parent folder.
        project_name : str             (required, keyword‑only)
            Sub‑folder + file naming stem.
        precision : int, default 6
            Decimal places for Begin/End times.
        annotation_label : str, default "Human"
            Value for the Raven “Annotation” column.
        low_freq / high_freq : int
            Frequency bounds (Hz) drawn on Raven’s spectrogram box.
        """

        base_dir = Path(base_dir)
        out_root = base_dir / "Raven Outputs" / project_name
        out_root.mkdir(parents=True, exist_ok=True)

        # ----- quick sanity check --------------------------------------
        req = {"file_path", "file_name", "start_time", "end_time"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(
                f"RavenTxtTransform: DataFrame missing column(s): {missing}"
            )

        # Work on a *copy* so we can mutate safely
        df = df.copy()

        # Absolute paths for each detection
        df["abs_path"] = df.apply(
            lambda r: str(Path(r.file_path) / r.file_name), axis=1
        )

        # ----------------------------------------------------------------
        #  1) Build the listfile – unique WAVs in first‑appearance order
        # ----------------------------------------------------------------
        unique_paths_in_order = pd.unique(df["abs_path"])
        listfile_path = out_root / f"{project_name}_listfile.txt"
        listfile_path.write_text("\n".join(unique_paths_in_order) + "\n")

        # ----------------------------------------------------------------
        #  2) Measure each WAV’s duration once
        # ----------------------------------------------------------------
        durations: Dict[str, float] = {}
        cumulative_offset: Dict[str, float] = {}

        running_total = 0.0
        for wav_path in unique_paths_in_order:
            try:
                dur = _wav_duration(wav_path)
            except Exception:
                # Fallback: use the largest DETECTION end‑time in that file
                dur = (
                    df.loc[df["abs_path"] == wav_path, "end_time"]
                    .max(skipna=True)
                    .item()
                )
            durations[wav_path] = dur
            cumulative_offset[wav_path] = running_total
            running_total += dur

        # ----------------------------------------------------------------
        #  3) Construct the Raven results table
        # ----------------------------------------------------------------
        df["Begin Time (s)"] = (
            df.apply(
                lambda r: cumulative_offset[r.abs_path] + float(r.start_time),
                axis=1,
            ).round(precision)
        )
        df["End Time (s)"] = (
            df.apply(
                lambda r: cumulative_offset[r.abs_path] + float(r.end_time),
                axis=1,
            ).round(precision)
        )

        results = pd.DataFrame({
            "Selection":      range(1, len(df) + 1),
            "View":           "Spectrogram 1",
            "Channel":        1,
            "Begin Time (s)": df["Begin Time (s)"],
            "End Time (s)":   df["End Time (s)"],
            "Low Freq (Hz)":  low_freq,
            "High Freq (Hz)": high_freq,
            "Annotation":     annotation_label,
            "Begin Path":     df["abs_path"],
            # -------- extra / editable columns -------------------------
            "erase":           df.get("erase", pd.Series([""] * len(df))),
            "user_comment":    df.get("user_comment", pd.Series([""] * len(df))),
            "review_datetime": df.get("review_datetime", pd.Series([""] * len(df))),
        })

        # Confidence can be appended automatically if it exists
        if "confidence" in df.columns:
            results["confidence"] = df["confidence"]

        # ----------------------------------------------------------------
        #  4) Write the results table (tab‑delimited, Windows line‑ends OK)
        # ----------------------------------------------------------------
        results_path = out_root / f"{project_name}.txt"
        results.to_csv(results_path, sep="\t", index=False, lineterminator="\n")

        # Done – return None → manager knows we handled persistence
        return None

