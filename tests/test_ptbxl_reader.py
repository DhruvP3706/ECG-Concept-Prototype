"""Synthetic tests for the WFDB-free PTB-XL reader. Run: .venv\\Scripts\\python.exe -I tests\\test_ptbxl_reader.py"""
import os
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept import ptbxl_reader as rd  # noqa: E402

FILE_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NAME = "00001_hr"


def make_channels(seed=0):
    return np.random.RandomState(seed).randint(-32768, 32768, size=(12, 5000)).astype(np.int16)   # full range: exercises checksum wrap


def header_text(ch, names=FILE_ORDER, name=NAME, unsigned_chk=False, **over):
    lines = [over.get("rec", f"{name} 12 500 5000")]
    for j, n in enumerate(names):
        chk = int(ch[j].astype(np.int64).sum()) % 65536
        chk = chk if unsigned_chk else (chk - 65536 if chk >= 32768 else chk)
        lines.append(over.get("sig", "{f}.dat {fmt} {g}({b})/{u} {r} {z} {i} {c} 0 {n}").format(
            f=name, fmt=over.get("fmt", "16"), g=over.get("gain", "1000.0"), b=over.get("base", 0), u=over.get("units", "mV"),
            r=16, z=0, i=int(ch[j][0]) + (1 if over.get("bad_init") == j else 0), c=chk + (1 if over.get("bad_chk") == j else 0), n=n))
    return "\n".join(lines) + "\n"


def write_record(d, ch, text=None, raw=None, name=NAME, **kw):
    base = os.path.join(d, name)
    with open(base + ".hea", "w", newline="") as f:
        f.write(text if text is not None else header_text(ch, name=name, **kw))
    with open(base + ".dat", "wb") as f:
        f.write(raw if raw is not None else ch.T.astype("<i2").tobytes())      # (T, 12) C-order = sample-major interleaving
    return base


class TestDecode(unittest.TestCase):
    def test_roundtrip_little_endian_sample_major_and_header_facts(self):
        ch = make_channels()
        with tempfile.TemporaryDirectory() as d:
            sig, h = rd.read_record(write_record(d, ch))
        self.assertEqual((sig.shape, sig.dtype), ((12, 5000), np.int16))
        self.assertTrue(np.array_equal(sig, ch))                                  # file order == model order after name reorder
        self.assertEqual((h["fs"], h["sig_len"], h["gain"], h["baseline"], h["units"], h["fmt"]), (500, 5000, 1000.0, 0, "mV", 16))
        self.assertEqual(h["original_lead_names"], FILE_ORDER)

    def test_little_endian_bytes(self):
        t = 5000
        ch = np.zeros((12, t), np.int16)
        ch[0, :] = 0x0102
        ch[1, :] = -2
        raw = ch.T.astype("<i2").tobytes()
        self.assertEqual(raw[:4], bytes([0x02, 0x01, 0xFE, 0xFF]))               # sample 0: lead0 low byte first, then lead1 (-2)
        d = rd.decode_samples(raw, rd.parse_header(header_text(ch), NAME))
        self.assertEqual((int(d[0, 0]), int(d[0, 1])), (0x0102, -2))

    def test_multiplexing_is_sample_major(self):
        ch = np.arange(12 * 5000, dtype=np.int32).reshape(12, 5000).astype(np.int16)
        raw = ch.T.astype("<i2").tobytes()
        flat = np.frombuffer(raw, "<i2")
        self.assertEqual(int(flat[1]), int(ch[1, 0]))                              # second value in the file is lead 1, sample 0
        self.assertEqual(int(flat[12]), int(ch[0, 1]))                             # value 12 is lead 0, sample 1
        d = rd.decode_samples(raw, rd.parse_header(header_text(ch), NAME))
        self.assertTrue(np.array_equal(d.T, ch))

    def test_reorder_by_name_from_shuffled_header(self):
        ch = make_channels(1)
        names = ["V6", "AVL", "I", "V2", "AVR", "III", "II", "V1", "V5", "AVF", "V4", "V3"]
        with tempfile.TemporaryDirectory() as d:
            sig, _ = rd.read_record(write_record(d, ch, names=names))
        for j, n in enumerate(px.MODEL_LEAD_ORDER):
            self.assertTrue(np.array_equal(sig[j], ch[names.index(n.upper())]), n)

    def test_checksum_signed_or_unsigned_header_both_accepted(self):
        ch = make_channels(2)
        for u in (False, True):
            with tempfile.TemporaryDirectory() as d:
                sig, _ = rd.read_record(write_record(d, ch, unsigned_chk=u))
            self.assertTrue(np.array_equal(sig, ch))


class TestRejections(unittest.TestCase):
    def fails(self, regex, **kw):
        ch = kw.pop("ch", None)
        ch = make_channels(3) if ch is None else ch
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(rd.RecordError, regex):
                rd.read_record(write_record(d, ch, **kw))

    def test_bad_checksum_names_the_channel(self):
        self.fails(r"V1: checksum", bad_chk=6)

    def test_bad_initial_value_names_the_channel(self):
        self.fails(r"II: initial value", bad_init=1)

    def test_checksum_detects_corrupted_sample(self):
        ch = make_channels(4)
        raw = bytearray(ch.T.astype("<i2").tobytes())
        raw[2 * (100 * 12 + 3)] ^= 0x01                                           # flip one bit in lead AVR, sample 100
        self.fails(r"AVR: checksum", ch=ch, raw=bytes(raw))

    def test_wrong_byte_count(self):
        ch = make_channels()
        raw = ch.T.astype("<i2").tobytes()
        self.fails(r"bytes, expected 120000", ch=ch, raw=raw[:-2])
        self.fails(r"bytes, expected 120000", ch=ch, raw=raw + b"\x00\x00")

    def test_wrong_format_rate_dimensions_and_units(self):
        self.fails("format", fmt="212")
        self.fails("unsupported", fmt="16x2")
        self.fails("unsupported", fmt="16+24")
        self.fails("unsupported", rec=f"{NAME} 12 100 5000")
        self.fails("unsupported", rec=f"{NAME} 12 500 1000")
        self.fails("unsupported", rec=f"{NAME} 8 500 5000")
        self.fails("unsupported", gain="200.0")
        self.fails("unsupported", base=5)
        self.fails("unsupported", units="uV")

    def test_lead_problems(self):
        dup = FILE_ORDER[:11] + ["I"]
        self.fails(r"duplicate leads", names=dup)
        self.fails(r"unexpected lead name", names=FILE_ORDER[:11] + ["V7"])
        self.fails(r"missing leads|duplicate", names=FILE_ORDER[:11] + ["i"])

    def test_corrupt_or_malformed_header(self):
        ch = make_channels()
        self.fails("corrupt|malformed|signal lines", text="")
        self.fails("non-printable|corrupt", text="x|\x00\x01 garbage\n" * 13)
        self.fails("malformed", text="not a header\n")
        good = header_text(ch).splitlines()
        self.fails("signal lines", text="\n".join(good[:12]) + "\n")                 # 11 signal lines
        self.fails("malformed signal line", text="\n".join(good[:5] + ["junk line"] + good[6:]) + "\n")
        self.fails("record name", text=header_text(ch).replace(NAME + " 12", "99999_hr 12", 1))
        self.fails("single-file", text=header_text(ch).replace(NAME + ".dat", "other.dat", 1))

    def test_non_ascii_header_bytes(self):
        ch = make_channels()
        with tempfile.TemporaryDirectory() as d:
            base = write_record(d, ch)
            with open(base + ".hea", "wb") as f:
                f.write(bytes(range(128, 200)) * 10)
            with self.assertRaisesRegex(rd.RecordError, "not ASCII"):
                rd.read_record(base)

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                rd.read_record(os.path.join(d, NAME))


class TestFoldGuard(unittest.TestCase):
    def test_fold10_blocked_before_any_file_access(self):
        r = px.PtbxlRecord(5, 5, 10, "records500/00000/00005_hr", "x", {"NORM": 100.0})
        with unittest.mock.patch("builtins.open", side_effect=AssertionError("file opened")), \
                unittest.mock.patch("ecg_concept.ptbxl_reader.read_record", side_effect=AssertionError("reader called")):
            with self.assertRaises(PermissionError):
                px.load_signal_native(r, "does-not-matter")

    def test_train_val_records_load_and_test_needs_explicit_flag(self):
        ch = make_channels(5)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "records500", "00000"))
            write_record(os.path.join(d, "records500", "00000"), ch, name="00007_hr")
            for fold, allow in ((1, False), (9, False), (10, True)):
                r = px.PtbxlRecord(7, 7, fold, "records500/00000/00007_hr", "x", {"NORM": 100.0})
                ls = px.load_signal_native(r, d, allow_test=allow)
                self.assertEqual((ls.signal.shape, ls.signal.dtype), ((12, 5000), np.int16))


if __name__ == "__main__":
    unittest.main(verbosity=1)
