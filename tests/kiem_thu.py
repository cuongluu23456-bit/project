"""Bộ kiểm thử tự động. Chạy từ thư mục gốc của dự án:

    python tests/kiem_thu.py

Các kiểm thử tập trung vào những chỗ dễ sai nhất của dự án:
  * công thức chi phí và bảng tra phải khớp nhau;
  * Quy hoạch động phải đúng như vét cạn trên bài toán nhỏ;
  * mọi thuật toán phải trả về phương án HỢP LỆ;
  * dự báo không được "nhìn trộm" nhu cầu của chính ngày cần dự báo (rò rỉ dữ liệu);
  * chia dữ liệu theo thời gian, kịch bản nhu cầu hợp lệ, script UCI chạy đúng, demo chạy trọn vẹn.
"""
import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC))

import chuan_bi_du_lieu_uci  # noqa: E402
import du_bao  # noqa: E402
import du_lieu  # noqa: E402
from bai_toan import BaiToan, tao_bai_toan, tao_thong_so_mat_hang  # noqa: E402
from di_truyen import thuat_toan_di_truyen  # noqa: E402
from quy_hoach_dong import quy_hoach_dong  # noqa: E402
from tham_lam import CAC_CACH_TINH, tham_lam  # noqa: E402


def tao_bai_toan_ngau_nhien(n=8, so_kich_ban=60, ti_le=0.5, hat=0, co_phi_co_dinh=True):
    """Một bài toán nhỏ, ngẫu nhiên, dùng chung cho nhiều kiểm thử."""
    rng = np.random.default_rng(hat)
    gia_ban = rng.uniform(2, 10, n)
    nhu_cau_tb = rng.uniform(5, 40, n)
    thong_so = tao_thong_so_mat_hang(gia_ban, nhu_cau_tb, hat=hat)
    if not co_phi_co_dinh:
        thong_so["phi_co_dinh"] = np.zeros(n)
        thong_so["quy_cach"] = np.ones(n, dtype=int)
    kich_ban = np.rint(rng.gamma(4, nhu_cau_tb / 4, size=(so_kich_ban, n)))
    return tao_bai_toan(thong_so, kich_ban, nhu_cau_tb, ti_le, ti_le)


class KiemThuBaiToan(unittest.TestCase):
    def test_bang_chi_phi_khop_cong_thuc_truc_tiep(self):
        """Chi phí tra bảng phải bằng chi phí tính thẳng từ công thức Q(q) trong README."""
        bt = tao_bai_toan_ngau_nhien(hat=1)
        rng = np.random.default_rng(5)
        for _ in range(20):
            so_thung = rng.integers(0, bt.so_thung_toi_da + 1)
            q = bt.so_luong(so_thung)
            truc_tiep = (bt.gia_nhap * q).sum() + (bt.phi_co_dinh * (q > 0)).sum()
            for kich_ban in bt.kich_ban:  # trung bình trên các kịch bản
                truc_tiep += (bt.phi_ton_kho * np.maximum(q - kich_ban, 0)).sum() / len(bt.kich_ban)
                truc_tiep += (bt.gia_ban * np.maximum(kich_ban - q, 0)).sum() / len(bt.kich_ban)
            self.assertAlmostEqual(bt.chi_phi(so_thung), truc_tiep, places=6)

    def test_nghiem_nguoi_ban_bao_newsvendor(self):
        """Không ràng buộc, không phí cố định, quy cách 1 -> nhập tối ưu = phân vị (P-c)/(P+h) của nhu cầu."""
        n = 6
        rng = np.random.default_rng(3)
        thong_so = tao_thong_so_mat_hang(rng.uniform(2, 10, n), np.full(n, 20.0))
        thong_so["phi_co_dinh"] = np.zeros(n)
        thong_so["quy_cach"] = np.ones(n, dtype=int)
        kich_ban = np.rint(rng.gamma(4, 5, size=(400, n)))
        bt = BaiToan(**thong_so, kich_ban=kich_ban, ngan_sach=1e12, suc_chua=1e12)
        for i in range(n):
            ti_le_toi_gioi = (bt.gia_ban[i] - bt.gia_nhap[i]) / (bt.gia_ban[i] + bt.phi_ton_kho[i])
            q_ly_thuyet = np.quantile(kich_ban[:, i], ti_le_toi_gioi, method="inverted_cdf")
            q_tot_nhat = int(np.argmin(bt.bang_chi_phi[i, :bt.so_thung_toi_da[i] + 1]))
            chi_phi_ly_thuyet = bt.bang_chi_phi[i, int(q_ly_thuyet)]
            self.assertAlmostEqual(bt.bang_chi_phi[i, q_tot_nhat], chi_phi_ly_thuyet, places=6)

    def test_chi_phi_thuc_te_va_ti_le_dap_ung(self):
        bt = tao_bai_toan_ngau_nhien(n=3, hat=2, co_phi_co_dinh=False)
        so_thung = np.array([2, 0, 1])
        q = bt.so_luong(so_thung)
        nhu_cau = np.array([float(q[0] + 3), 5.0, float(max(q[2] - 1, 0))])
        kq = bt.chi_phi_thuc_te(so_thung, nhu_cau)
        thieu = bt.gia_ban[0] * 3 + bt.gia_ban[1] * 5
        du = bt.phi_ton_kho[2] * (1 if q[2] >= 1 else 0)
        nhap = float((bt.gia_nhap * q).sum())
        self.assertAlmostEqual(kq["tong_chi_phi"], nhap + du + thieu, places=6)
        self.assertEqual(kq["so_mat_hang_thieu"], 2)
        self.assertAlmostEqual(kq["ti_le_dap_ung"], (q[0] + 0 + min(q[2], nhu_cau[2])) / nhu_cau.sum(), places=6)

    def test_sua_nghiem_co_ty_le_luon_hop_le(self):
        bt = tao_bai_toan_ngau_nhien(ti_le=0.3, hat=4)
        rng = np.random.default_rng(1)
        quan_the = rng.integers(0, bt.so_thung_toi_da + 1, size=(50, bt.n))
        da_sua = bt.sua_nghiem_co_ty_le(quan_the)
        for hang in da_sua:
            self.assertTrue(bt.hop_le(hang))
        self.assertTrue(bt.hop_le(bt.sua_nghiem_tham_lam(bt.so_thung_toi_da)))


class KiemThuThuatToan(unittest.TestCase):
    def test_quy_hoach_dong_bang_vet_can_tren_bai_toan_nho(self):
        """Với dữ liệu nguyên và lưới đúng bằng ngân sách/kho, DP phải ra CHÍNH XÁC nghiệm vét cạn."""
        for hat in range(12):
            rng = np.random.default_rng(hat)
            n = 4
            gia_nhap = rng.integers(1, 5, n).astype(float)
            thong_so = dict(
                gia_ban=gia_nhap * rng.uniform(1.8, 3.0, n), gia_nhap=gia_nhap,
                the_tich=rng.integers(1, 4, n).astype(float), phi_ton_kho=0.1 * gia_nhap,
                phi_co_dinh=rng.integers(0, 4, n).astype(float), quy_cach=rng.integers(1, 3, n))
            kich_ban = rng.integers(0, 9, size=(30, n)).astype(float)
            ngan_sach, suc_chua = int(rng.integers(6, 16)), int(rng.integers(6, 16))
            bt = BaiToan(**thong_so, kich_ban=kich_ban, ngan_sach=ngan_sach, suc_chua=suc_chua)
            tot_nhat = min(
                (bt.chi_phi(np.array(g)) for g in itertools.product(*[range(m + 1) for m in bt.so_thung_toi_da])
                 if bt.hop_le(np.array(g))))
            g_dp = quy_hoach_dong(bt, so_o_ngan_sach=ngan_sach, so_o_kho=suc_chua)
            self.assertTrue(bt.hop_le(g_dp))
            self.assertAlmostEqual(bt.chi_phi(g_dp), tot_nhat, places=6, msg=f"bài toán số {hat}")

    def test_moi_thuat_toan_tra_ve_phuong_an_hop_le_va_khong_te_hon_khong_nhap(self):
        for ti_le in (0.2, 0.5, 1.0):
            bt = tao_bai_toan_ngau_nhien(ti_le=ti_le, hat=7)
            khong_nhap = bt.chi_phi(np.zeros(bt.n, dtype=int))
            cac_phuong_an = {"dp": quy_hoach_dong(bt, 150, 100), "ga_sua": thuat_toan_di_truyen(bt, so_the_he=60, cach_rang_buoc="sua")[0],
                             "ga_phat": thuat_toan_di_truyen(bt, so_the_he=60, cach_rang_buoc="phat")[0]}
            for cach in CAC_CACH_TINH:
                cac_phuong_an[f"tham_lam_{cach}"] = tham_lam(bt, cach)
            for ten, g in cac_phuong_an.items():
                self.assertTrue(bt.hop_le(g), f"{ten} vi phạm ràng buộc ở tỉ lệ {ti_le}")
                self.assertTrue(((g >= 0) & (g <= bt.so_thung_toi_da)).all(), f"{ten} có số thùng ngoài giới hạn")
                self.assertLessEqual(bt.chi_phi(g), khong_nhap + 1e-9, f"{ten} tệ hơn việc không nhập gì")

    def test_ga_lap_lai_duoc_va_khong_tra_ket_qua_xau_di(self):
        bt = tao_bai_toan_ngau_nhien(hat=9)
        g1, ls1 = thuat_toan_di_truyen(bt, kich_thuoc_quan_the=40, so_the_he=50, hat=3)
        g2, ls2 = thuat_toan_di_truyen(bt, kich_thuoc_quan_the=40, so_the_he=50, hat=3)
        np.testing.assert_array_equal(g1, g2)
        self.assertEqual(ls1, ls2)
        # ưu tú được giữ lại -> độ thích nghi tốt nhất không bao giờ tăng qua các thế hệ
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(ls1, ls1[1:])))

    def test_ga_gan_bang_dp_tren_bai_toan_vua(self):
        bt = tao_bai_toan_ngau_nhien(n=10, ti_le=0.5, hat=11)
        chi_phi_dp = bt.chi_phi(quy_hoach_dong(bt, 300, 200))
        chi_phi_ga = np.mean([bt.chi_phi(thuat_toan_di_truyen(bt, hat=h)[0]) for h in range(3)])
        self.assertLess(chi_phi_ga, chi_phi_dp * 1.02)

    def test_tham_lam_gap_phi_co_dinh_van_hop_le(self):
        """Khi có phí cố định, Greedy có thể bỏ lỡ mặt hàng nhưng không bao giờ được phép vi phạm ràng buộc."""
        bt = tao_bai_toan_ngau_nhien(ti_le=0.4, hat=13, co_phi_co_dinh=True)
        self.assertTrue(bt.hop_le(tham_lam(bt)))


class KiemThuDuBao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.du_lieu = du_lieu.sinh_du_lieu_gia(so_mat_hang=4, so_ngay=150)

    def test_khong_ro_ri_du_lieu_cua_chinh_ngay_du_bao(self):
        """Đổi doanh số của MỘT ngày thành số rất lớn: đặc trưng của ngày đó không được thay đổi."""
        goc = du_bao.tao_dac_trung(self.du_lieu)
        ngay = np.sort(goc["ngay"].unique())[60]
        sua = self.du_lieu.copy()
        sua.loc[sua["ngay"] == ngay, "so_luong"] = 10 ** 6
        moi = du_bao.tao_dac_trung(sua)
        a = goc[goc["ngay"] == ngay].sort_values("ma_hang")[du_bao.COT_DAC_TRUNG].to_numpy()
        b = moi[moi["ngay"] == ngay].sort_values("ma_hang")[du_bao.COT_DAC_TRUNG].to_numpy()
        np.testing.assert_allclose(a, b)
        # còn các ngày SAU đó thì được phép thay đổi (vì doanh số ngày đó đã là quá khứ của chúng)
        sau = np.sort(goc["ngay"].unique())[61]
        a2 = goc[goc["ngay"] == sau]["lag_1"].to_numpy()
        b2 = moi[moi["ngay"] == sau]["lag_1"].to_numpy()
        self.assertFalse(np.allclose(a2, b2))

    def test_chia_theo_thoi_gian_dung_thu_tu_va_khong_giao_nhau(self):
        train, valid, test = du_bao.chia_theo_thoi_gian(du_bao.tao_dac_trung(self.du_lieu))
        self.assertLess(train["ngay"].max(), valid["ngay"].min())
        self.assertLess(valid["ngay"].max(), test["ngay"].min())
        self.assertGreater(len(train), len(valid))
        self.assertGreater(len(valid), 0)
        self.assertGreater(len(test), 0)

    def test_kich_ban_nhu_cau_la_so_nguyen_khong_am(self):
        phan_du = [np.array([-5.0, -1.0, 0.0, 2.0, 7.0])] * 3
        mu = np.array([0.5, 10.0, 100.0])
        kb = du_bao.tao_kich_ban_nhu_cau(mu, phan_du, 500, np.random.default_rng(0))
        self.assertEqual(kb.shape, (500, 3))
        self.assertTrue((kb >= 0).all())
        np.testing.assert_array_equal(kb, np.rint(kb))
        self.assertAlmostEqual(kb[:, 2].mean(), 100 + np.mean(phan_du[2]), delta=1.5)

    def test_du_bao_ngay_mai_bo_qua_ngay_dong_cua(self):
        ngay_cuoi = pd.Timestamp("2011-03-04")  # thứ Sáu
        self.assertEqual(du_bao.ngay_mo_cua_ke_tiep(ngay_cuoi, []), pd.Timestamp("2011-03-05"))      # thứ Bảy
        self.assertEqual(du_bao.ngay_mo_cua_ke_tiep(ngay_cuoi, [5]), pd.Timestamp("2011-03-06"))     # đóng thứ Bảy -> Chủ nhật
        self.assertEqual(du_bao.ngay_mo_cua_ke_tiep(ngay_cuoi, [5, 6]), pd.Timestamp("2011-03-07"))  # đóng cả cuối tuần -> thứ Hai

    def test_mo_hinh_ml_thang_co_so_tren_du_lieu_co_quy_luat(self):
        du_lieu_lon = du_lieu.sinh_du_lieu_gia(so_mat_hang=8, so_ngay=330)
        train, valid, test = du_bao.chia_theo_thoi_gian(du_bao.tao_dac_trung(du_lieu_lon))
        bang, _, _ = du_bao.so_sanh_cac_mo_hinh(train, valid, test)
        mae = bang.set_index("Phương pháp")["MAE (test)"]
        self.assertLess(mae["Random Forest"], mae[du_bao.TEN_TUAN_TRUOC])


class KiemThuDuLieu(unittest.TestCase):
    def test_du_lieu_gia_dung_dinh_dang_va_dong_cua_thu_bay(self):
        d = du_lieu.sinh_du_lieu_gia(so_mat_hang=3, so_ngay=60)
        self.assertEqual(list(d.columns), du_lieu.CAC_COT_BAT_BUOC)
        self.assertEqual(du_lieu.tim_thu_dong_cua(d), [5])
        self.assertTrue((d["so_luong"] >= 0).all() and (d["don_gia"] > 0).all())

    def test_doc_du_lieu_bao_loi_khi_thieu_cot(self):
        with tempfile.TemporaryDirectory() as thu_muc:
            duong_dan = Path(thu_muc) / "sai.csv"
            pd.DataFrame({"ngay": ["2011-01-03"], "ma_hang": ["A"]}).to_csv(duong_dan, index=False)
            with self.assertRaises(ValueError):
                du_lieu.doc_du_lieu(duong_dan)


class KiemThuChuanBiUCI(unittest.TestCase):
    def test_lam_sach_va_tao_bang_du_so_ngay(self):
        """File giả đúng định dạng UCI: có hóa đơn hủy (C), số lượng âm, mã không phải sản phẩm, mã hóa latin-1."""
        dong = []
        for ngay in pd.date_range("2011-01-03", periods=10, freq="D"):
            if ngay.weekday() == 5:
                continue
            for ma, so_luong in (("10001", 5), ("10002", 3), ("POST", 1)):
                dong.append((f"5{ngay.day:02d}{ma}", ma, "Café mau " + ma, so_luong, ngay + pd.Timedelta(hours=10), 2.5, 1))
        dong.append(("C599999", "10001", "Hoa don huy (Café)", -4, pd.Timestamp("2011-01-05 09:00"), 2.5, 1))
        dong.append(("599998", "10002", "Gia bang 0 (Café)", 2, pd.Timestamp("2011-01-05 09:00"), 0.0, 1))
        bang = pd.DataFrame(dong, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"])
        with tempfile.TemporaryDirectory() as thu_muc:
            vao, ra = Path(thu_muc) / "uci.csv", Path(thu_muc) / "ra" / "ban_hang.csv"
            bang.to_csv(vao, index=False, encoding="latin-1")  # "é" mã hóa được bằng latin-1 như file UCI thật
            ket_qua = chuan_bi_du_lieu_uci.chuan_bi(vao, ra, so_mat_hang=5)
            doc_lai = du_lieu.doc_du_lieu(ra)
        self.assertEqual(set(ket_qua["ma_hang"]), {"10001", "10002"})  # "POST" bị loại vì không phải mã sản phẩm
        self.assertEqual(ket_qua["ngay"].nunique(), 9)
        self.assertEqual(len(ket_qua), 9 * 2)  # lưới đầy đủ ngày × mặt hàng
        self.assertTrue((ket_qua["so_luong"] >= 0).all())
        self.assertTrue((ket_qua["don_gia"] > 0).all())
        self.assertEqual(len(doc_lai), len(ket_qua))


class KiemThuChayDemo(unittest.TestCase):
    def test_demo_chay_tron_ven_che_do_nhanh(self):
        with tempfile.TemporaryDirectory() as thu_muc:
            ket_qua = subprocess.run(
                [sys.executable, str(GOC / "chay_demo.py"), "--nhanh", "--so-hat-ga", "2", "--so-ngay-backtest", "3",
                 "--ket-qua", thu_muc], capture_output=True, text=True, cwd=GOC, timeout=300)
            self.assertEqual(ket_qua.returncode, 0, ket_qua.stderr[-2000:])
            tep = {p.name for p in Path(thu_muc).iterdir()}
            for ten in ("bang_E1_du_bao.csv", "ke_hoach_ngay_mai.csv", "bang_E2_toi_uu_mot_ngay.csv",
                        "bang_E3_backtest_tom_tat.csv", "hinh_1_du_bao.png", "hinh_5_backtest.png"):
                self.assertIn(ten, tep)
        self.assertIn("DỮ LIỆU GIẢ", ket_qua.stdout)  # phải cảnh báo rõ khi dùng dữ liệu giả


if __name__ == "__main__":
    unittest.main(verbosity=2)
