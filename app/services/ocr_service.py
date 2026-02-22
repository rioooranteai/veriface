from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger()


class JenisKelamin(str, Enum):
    LAKI_LAKI = "LAKI-LAKI"
    PEREMPUAN = "PEREMPUAN"


class StatusPerkawinan(str, Enum):
    BELUM_KAWIN = "BELUM KAWIN"
    KAWIN = "KAWIN"
    CERAI_HIDUP = "CERAI HIDUP"
    CERAI_MATI = "CERAI MATI"


class Kewarganegaraan(str, Enum):
    WNI = "WNI"
    WNA = "WNA"


AGAMA_VALID: frozenset[str] = frozenset(
    ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "BUDHA", "KONGHUCU", "KEPERCAYAAN"]
)

OCR_DIGIT_FIX: dict[str, str] = {
    "O": "0", "o": "0",
    "I": "1", "l": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

_RE = {
    "nik_valid": re.compile(r"^\d{16}$"),
    "trig_nama": re.compile(r"\bnama\b", re.IGNORECASE),
    "trig_ttl": re.compile(r"\b(tempat|tgl|tanggal|lahir)\b", re.IGNORECASE),
    "trig_kelamin": re.compile(r"\b(jenis\s*kelamin|kelamin)\b", re.IGNORECASE),
    "trig_darah": re.compile(r"\bgol(ongan)?\s*(darah)?\b", re.IGNORECASE),
    "trig_alamat": re.compile(r"^alamat\b", re.IGNORECASE),
    "trig_rtrw": re.compile(r"\brt\s*/?\s*rw\b", re.IGNORECASE),
    "trig_kel": re.compile(r"^(kel(urahan)?|desa)\b", re.IGNORECASE),
    "trig_kec": re.compile(r"^kecamatan\b", re.IGNORECASE),
    "trig_agama": re.compile(r"^agama\b", re.IGNORECASE),
    "trig_status": re.compile(r"\bstatus\b", re.IGNORECASE),
    "trig_pek": re.compile(r"\bpekerjaan\b", re.IGNORECASE),
    "trig_warga": re.compile(r"\b(kewarganegaraan|warga)\b", re.IGNORECASE),
    "trig_berlaku": re.compile(r"\bberlaku\b", re.IGNORECASE),
    "val_ttl": re.compile(r"([A-Z][A-Z\s\-]+?),?\s*(\d{2}[-/]\d{2}[-/]\d{4})", re.IGNORECASE),
    "val_kelamin": re.compile(r"(LAKI\s*-\s*LAKI|PEREMPUAN)", re.IGNORECASE),
    "val_gol_darah": re.compile(r"\b(AB|A|B|O)\b"),
    "val_rtrw": re.compile(r"(\d{1,3})\s*/\s*(\d{1,3})"),
    "val_status": re.compile(r"(BELUM\s+KAWIN|CERAI\s+HIDUP|CERAI\s+MATI|KAWIN)", re.IGNORECASE),
    "val_warga": re.compile(r"\b(WNI|WNA)\b", re.IGNORECASE),
    "val_berlaku": re.compile(r"(\d{2}[-/]\d{2}[-/]\d{4}|SEUMUR\s+HIDUP)", re.IGNORECASE),
    "val_date": re.compile(r"\d{2}[-/]\d{2}[-/]\d{4}"),
    "strip_label": re.compile(
        r"^(nama|alamat|kecamatan|pekerjaan|kewarganegaraan|agama|"
        r"kel(urahan)?(/desa)?|desa|rt\s*/?\s*rw|berlaku\s*(hingga)?)"
        r"[\s:./\-]*",
        re.IGNORECASE
    ),
}

_ALL_FIELDS: frozenset[str] = frozenset([
    "nik", "nama", "tempat_lahir", "tgl_lahir",
    "jenis_kelamin", "gol_darah", "alamat", "rt_rw",
    "kelurahan", "kecamatan", "agama", "status_perkawinan",
    "pekerjaan", "kewarganegaraan", "berlaku_hingga",
])


@dataclass
class KTPData:
    nik: Optional[str] = None
    nama: Optional[str] = None
    tempat_lahir: Optional[str] = None
    tgl_lahir: Optional[str] = None
    jenis_kelamin: Optional[str] = None
    gol_darah: Optional[str] = None
    alamat: Optional[str] = None
    rt_rw: Optional[str] = None
    kelurahan: Optional[str] = None
    kecamatan: Optional[str] = None
    agama: Optional[str] = None
    status_perkawinan: Optional[str] = None
    pekerjaan: Optional[str] = None
    kewarganegaraan: Optional[str] = None
    berlaku_hingga: Optional[str] = None
    confidence_avg: float = 0.0
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    @property
    def completeness(self) -> float:
        filled = sum(1 for f in _ALL_FIELDS if getattr(self, f) is not None)
        return filled / len(_ALL_FIELDS)

    def is_valid_nik(self) -> bool:
        return bool(self.nik and _RE["nik_valid"].match(self.nik))


class KTPOCRError(Exception):
    pass


class OCRPredictError(KTPOCRError):
    pass


def _fix_ocr_digit_noise(text: str) -> str:
    return "".join(OCR_DIGIT_FIX.get(c, c) for c in text)


def _strip_label(text: str) -> str:
    return _RE["strip_label"].sub("", text).strip()


def _next_nonempty(texts: list[str], idx: int, max_look: int = 2) -> Optional[str]:
    for j in range(idx + 1, min(idx + 1 + max_look, len(texts))):
        candidate = texts[j].strip()
        if candidate:
            return candidate
    return None


def _normalize_date(raw: str) -> str:
    return raw.replace("/", "-")


def _clean_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z\s'\-]", "", raw).strip()
    return " ".join(w.capitalize() for w in cleaned.split())


def _parse_ktp_texts(
        texts: list[str],
        scores: Optional[list[float]] = None,
        min_confidence: float = 0.65,
) -> KTPData:
    result = KTPData()
    warnings = result.parse_warnings
    found = set()

    if scores and len(scores) == len(texts):
        valid_pairs = [(t, s) for t, s in zip(texts, scores) if s >= min_confidence]
        if valid_pairs:
            texts, scores = zip(*valid_pairs)  # type: ignore
            texts = list(texts)
            scores = list(scores)
            result.confidence_avg = sum(scores) / len(scores)
        else:
            warnings.append("Semua skor OCR di bawah threshold.")
            return result
    else:
        result.confidence_avg = -1.0

    for i, raw_text in enumerate(texts):
        if len(found) >= len(_ALL_FIELDS):
            break

        text = raw_text.strip()
        upper = text.upper()

        if "nik" not in found:
            candidate = re.sub(r"[^0-9OolIlZSGB]", "", text)
            candidate = _fix_ocr_digit_noise(candidate)
            if _RE["nik_valid"].match(candidate):
                result.nik = candidate
                found.add("nik")

        if "nama" not in found and _RE["trig_nama"].search(upper):
            val = _strip_label(text)
            if not val:
                val = _next_nonempty(texts, i)
            if val:
                result.nama = _clean_name(val)
                found.add("nama")
            else:
                warnings.append(f"[{i}] Trigger 'Nama' ditemukan tapi nilai kosong.")

        if "tempat_lahir" not in found and _RE["trig_ttl"].search(upper):
            m = _RE["val_ttl"].search(text)
            if m:
                result.tempat_lahir = m.group(1).strip().title()
                result.tgl_lahir = _normalize_date(m.group(2).strip())
                found.update(["tempat_lahir", "tgl_lahir"])
            else:
                nxt = _next_nonempty(texts, i)
                if nxt:
                    m2 = _RE["val_ttl"].search(nxt)
                    if m2:
                        result.tempat_lahir = m2.group(1).strip().title()
                        result.tgl_lahir = _normalize_date(m2.group(2).strip())
                        found.update(["tempat_lahir", "tgl_lahir"])
                    else:
                        dm = _RE["val_date"].search(nxt)
                        if dm:
                            result.tgl_lahir = _normalize_date(dm.group())
                            found.add("tgl_lahir")
                if not m:
                    warnings.append(f"[{i}] Format TTL tidak cocok: {text!r}")

        if "jenis_kelamin" not in found:
            m = _RE["val_kelamin"].search(upper)
            if m:
                raw_jk = re.sub(r"\s+", " ", m.group(1)).upper().strip()
                raw_jk = re.sub(r"LAKI\s*-\s*LAKI", "LAKI-LAKI", raw_jk)
                result.jenis_kelamin = raw_jk
                found.add("jenis_kelamin")

        if "gol_darah" not in found and _RE["trig_darah"].search(upper):
            m = _RE["val_gol_darah"].search(upper)
            if m:
                result.gol_darah = m.group(1).upper()
                found.add("gol_darah")

        if "alamat" not in found and _RE["trig_alamat"].search(upper):
            val = _strip_label(text)
            if not val:
                val = _next_nonempty(texts, i)
            if val:
                result.alamat = val.title()
                found.add("alamat")

        if "rt_rw" not in found and _RE["trig_rtrw"].search(upper):
            m = _RE["val_rtrw"].search(text)
            if m:
                result.rt_rw = f"{m.group(1).zfill(3)}/{m.group(2).zfill(3)}"
                found.add("rt_rw")

        if "kelurahan" not in found and _RE["trig_kel"].search(upper):
            val = _strip_label(text)
            if not val:
                val = _next_nonempty(texts, i)
            if val:
                result.kelurahan = val.title()
                found.add("kelurahan")

        if "kecamatan" not in found and _RE["trig_kec"].search(upper):
            val = _strip_label(text)
            if not val:
                val = _next_nonempty(texts, i)
            if val:
                result.kecamatan = val.title()
                found.add("kecamatan")

        if "agama" not in found:
            if _RE["trig_agama"].search(upper):
                val = _strip_label(text).upper() or (_next_nonempty(texts, i) or "").upper()
            else:
                val = upper
            for agama in AGAMA_VALID:
                if agama in val:
                    result.agama = "BUDDHA" if agama == "BUDHA" else agama
                    found.add("agama")
                    break

        if "status_perkawinan" not in found:
            m = _RE["val_status"].search(upper)
            if m:
                result.status_perkawinan = re.sub(r"\s+", " ", m.group(1)).upper().strip()
                found.add("status_perkawinan")

        if "pekerjaan" not in found and _RE["trig_pek"].search(upper):
            val = _strip_label(text)
            if not val:
                val = _next_nonempty(texts, i)
            if val:
                result.pekerjaan = val.title()
                found.add("pekerjaan")

        if "kewarganegaraan" not in found and _RE["trig_warga"].search(upper):
            m = _RE["val_warga"].search(upper)
            if m:
                result.kewarganegaraan = m.group(1).upper()
                found.add("kewarganegaraan")

        if "berlaku_hingga" not in found and _RE["trig_berlaku"].search(upper):
            m = _RE["val_berlaku"].search(upper)
            if m:
                result.berlaku_hingga = re.sub(r"\s+", " ", m.group(1)).upper()
                found.add("berlaku_hingga")

    _post_validate(result)
    return result


def _post_validate(data: KTPData) -> None:
    w = data.parse_warnings

    if data.nik and not data.is_valid_nik():
        w.append(f"NIK tidak valid: {data.nik!r}")
        data.nik = None

    if data.nik and data.tgl_lahir:
        try:
            dd = int(data.nik[6:8])
            mm = int(data.nik[8:10])
            if dd > 40:
                dd -= 40
            if data.tgl_lahir[:5] != f"{dd:02d}-{mm:02d}":
                w.append(
                    f"Tanggal lahir di NIK tidak cocok dengan tgl_lahir ({data.tgl_lahir})"
                )
        except (ValueError, IndexError):
            pass

    if data.gol_darah and data.gol_darah not in {"A", "B", "AB", "O"}:
        w.append(f"Golongan darah tidak valid: {data.gol_darah!r}")
        data.gol_darah = None


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deskewed = _deskew(binary)

    return cv2.cvtColor(deskewed, cv2.COLOR_GRAY2RGB)


def _deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.5 or abs(angle) > 10:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class OCRService:

    def __init__(self, min_confidence: float = 0.65, debug: bool = False):
        self.min_confidence = min_confidence
        self.debug = debug
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=False,
            lang='id',
            enable_mkldnn=False
        )

    def extract_from_file(self, path: str) -> KTPData:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path!r}")
        return self.extract_from_array(image)

    def extract_from_array(self, image: np.ndarray) -> KTPData:
        t0 = time.perf_counter()

        preprocessed = _preprocess_image(image)
        texts, scores = self._run_ocr(preprocessed)

        if not texts:
            logger.warning("Tidak ada teks terdeteksi oleh OCR.")
            return KTPData(parse_warnings=["Tidak ada teks terdeteksi."])

        if self.debug:
            logger.debug("Teks OCR raw:\n%s", "\n".join(
                f"  [{i:02d}] ({s:.3f}) {t!r}"
                for i, (t, s) in enumerate(zip(texts, scores))
            ))

        result = _parse_ktp_texts(texts, scores, min_confidence=self.min_confidence)

        logger.info(
            "Ekstraksi selesai | %.3fs | completeness=%.0f%% | NIK=%s",
            time.perf_counter() - t0,
            result.completeness * 100,
            result.nik or "NOT FOUND",
        )

        for w in result.parse_warnings:
            logger.warning("⚠  %s", w)

        return result

    def _run_ocr(self, image: np.ndarray) -> tuple[list[str], list[float]]:
        try:
            raw = self.paddle_ocr.predict(image)
        except Exception as e:
            raise OCRPredictError(f"PaddleOCR predict gagal: {e}") from e

        if not raw:
            return [], []

        page = raw[0]
        texts = page.get("rec_texts", []) or []
        scores = page.get("rec_scores", []) or []
        min_len = min(len(texts), len(scores))
        return texts[:min_len], scores[:min_len]
