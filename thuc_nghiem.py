"""Các thực nghiệm đưa vào báo cáo.

  E1  Dự báo nhu cầu: Random Forest / Gradient Boosting có thắng các cơ sở đơn giản không?
  E2  Tối ưu cho MỘT ngày: so sánh các thuật toán ở nhiều mức ngân sách/kho (GA chạy nhiều seed).
  E3  Backtest nhiều ngày: lập kế hoạch rồi áp NHU CẦU THẬT để đo chi phí thực tế, tỉ lệ đáp ứng.
  Kế hoạch ngày mai: kết quả "sản phẩm" cho người ra quyết định.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import cau_hinh
import du_bao
from bai_toan import tao_bai_toan, tao_thong_so_mat_hang
from di_truyen import thuat_toan_di_truyen
from quy_hoach_dong import quy_hoach_dong
from tham_lam import CAC_CACH_TINH, tham_lam

TEN_THONG_THUONG = "Nhập thông thường"
TEN_THAM_LAM = "Tham lam"
TEN_GA = "Di truyền (GA)"
TEN_DP = "Quy hoạch động (DP)"
TEN_HOAN_HAO = "Dự báo hoàn hảo"
TEN_CACH_THAM_LAM = {"tong_chuan_hoa": "tổng chuẩn hóa", "ngan_sach": "theo ngân sách", "tuyet_doi": "tuyệt đối"}


@dataclass
class GoiChung:
    """Mọi thứ phần ML đã chuẩn bị, để các thực nghiệm dùng chung."""
    du_lieu: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    bang_du_bao: pd.DataFrame     # kết quả E1
    ten_mo_hinh: str              # mô hình ML tốt nhất theo tập valid
    mo_hinh: object
    phan_du: list                 # sai số dự báo trên valid, theo từng mặt hàng
    gia_ban_tb: np.ndarray
    nhu_cau_tb: np.ndarray
    thong_so: dict                # các tham số giả định (giá nhập, thể tích...)


def chuan_bi_chung(du_lieu):
    """Chạy phần ML một lần: đặc trưng, chia dữ liệu, huấn luyện, đo sai số, lập tham số mặt hàng."""
    dac_trung = du_bao.tao_dac_trung(du_lieu)
    train, valid, test = du_bao.chia_theo_thoi_gian(dac_trung)
    bang_du_bao, cac_mo_hinh, ten_tot_nhat = du_bao.so_sanh_cac_mo_hinh(train, valid, test)
    mo_hinh = cac_mo_hinh[ten_tot_nhat]
    gia_ban_tb = train.groupby("ma_hang_so")["don_gia"].mean().to_numpy()
    nhu_cau_tb = train.groupby("ma_hang_so")["so_luong"].mean().to_numpy()
    return GoiChung(du_lieu=du_lieu, train=train, valid=valid, test=test, bang_du_bao=bang_du_bao,
                    ten_mo_hinh=ten_tot_nhat, mo_hinh=mo_hinh, phan_du=du_bao.tinh_phan_du(mo_hinh, valid),
                    gia_ban_tb=gia_ban_tb, nhu_cau_tb=nhu_cau_tb,
                    thong_so=tao_thong_so_mat_hang(gia_ban_tb, nhu_cau_tb))


def hang_trong_ngay(bang, ngay):
    return bang[bang["ngay"] == ngay].sort_values("ma_hang_so")


def phuong_an_thong_thuong(bai_toan, nhu_cau_tuan_truoc):
    """Cách nhập quen thuộc: nhập bằng doanh số cùng thứ tuần trước (làm tròn theo thùng),
    nếu vượt ngân sách/kho thì cắt đồng loạt theo tỉ lệ."""
    so_thung = np.rint(np.asarray(nhu_cau_tuan_truoc, float) / bai_toan.quy_cach).astype(int)
    so_thung = np.clip(so_thung, 0, bai_toan.so_thung_toi_da)
    return bai_toan.sua_nghiem_co_ty_le(so_thung)


def _do_thoi_gian(ham):
    bat_dau = time.perf_counter()
    ket_qua = ham()
    return ket_qua, time.perf_counter() - bat_dau


# ====================================================================== KẾ HOẠCH NGÀY MAI
def lap_ke_hoach_ngay_mai(goi, ti_le_ngan_sach=cau_hinh.TI_LE_NGAN_SACH, ti_le_suc_chua=cau_hinh.TI_LE_SUC_CHUA):
    """Dự báo ngày mai -> sinh kịch bản -> giải bằng mọi thuật toán -> chọn phương án chi phí kỳ vọng thấp nhất."""
    ngay_mai, hang_mai, mu = du_bao.du_bao_ngay_mai(goi.du_lieu, goi.mo_hinh)
    rng = np.random.default_rng(cau_hinh.HAT_NGAU_NHIEN)
    kich_ban = du_bao.tao_kich_ban_nhu_cau(mu, goi.phan_du, cau_hinh.SO_KICH_BAN, rng)
    bai_toan = tao_bai_toan(goi.thong_so, kich_ban, goi.nhu_cau_tb, ti_le_ngan_sach, ti_le_suc_chua)

    cac_phuong_an, cac_giay = {}, {}
    cac_phuong_an[TEN_THONG_THUONG], cac_giay[TEN_THONG_THUONG] = _do_thoi_gian(
        lambda: phuong_an_thong_thuong(bai_toan, hang_mai["lag_7"].to_numpy()))
    cac_phuong_an[TEN_THAM_LAM], cac_giay[TEN_THAM_LAM] = _do_thoi_gian(lambda: tham_lam(bai_toan))
    ket_qua_ga, cac_giay[TEN_GA] = _do_thoi_gian(lambda: thuat_toan_di_truyen(bai_toan, hat=0))
    cac_phuong_an[TEN_GA] = ket_qua_ga[0]
    cac_phuong_an[TEN_DP], cac_giay[TEN_DP] = _do_thoi_gian(lambda: quy_hoach_dong(bai_toan))

    tong_ket = pd.DataFrame([{
        "Phương pháp": ten, "Chi phí kỳ vọng": bai_toan.chi_phi(g),
        "Tiền nhập / ngân sách (%)": 100 * bai_toan.tien_nhap(g) / bai_toan.ngan_sach,
        "Kho dùng / sức chứa (%)": 100 * bai_toan.the_tich_dung(g) / bai_toan.suc_chua,
        "Số mặt hàng nhập": int((g > 0).sum()), "Thời gian (giây)": cac_giay[ten],
    } for ten, g in cac_phuong_an.items()])
    de_xuat = min((t for t, g in cac_phuong_an.items() if bai_toan.hop_le(g)),
                  key=lambda t: bai_toan.chi_phi(cac_phuong_an[t]))

    xac_suat_thieu = bai_toan.xac_suat_thieu(cac_phuong_an[de_xuat])
    bang = pd.DataFrame({"Mã hàng": hang_mai["ma_hang"], "Tên hàng": hang_mai["ten_hang"], "Dự báo": mu.round(1)})
    for ten, g in cac_phuong_an.items():
        bang[ten] = bai_toan.so_luong(g)
    bang["Xác suất thiếu (%) – phương án đề xuất"] = (100 * xac_suat_thieu).round(0).astype(int)
    bang["Cảnh báo thiếu hàng"] = np.where(xac_suat_thieu >= cau_hinh.NGUONG_NGUY_CO_THIEU, "CÓ", "")
    return {"ngay": ngay_mai, "bang": bang, "tong_ket": tong_ket, "de_xuat": de_xuat,
            "bai_toan": bai_toan, "cac_phuong_an": cac_phuong_an}


# ====================================================================== E2: MỘT NGÀY, NHIỀU MỨC RÀNG BUỘC
def thuc_nghiem_toi_uu_mot_ngay(goi, cac_ti_le=(0.3, 0.5, 0.8, 1.5), so_hat=30):
    """So sánh các thuật toán trên cùng một bài toán (ngày đầu của tập test) ở nhiều mức ràng buộc.

    Ràng buộc càng chặt (tỉ lệ càng nhỏ) thì việc chọn mặt hàng nào càng quan trọng.
    GA chạy `so_hat` lần với các seed khác nhau -> báo cáo trung bình ± độ lệch chuẩn.
    """
    ngay = pd.Timestamp(np.sort(goi.test["ngay"].unique())[0])
    hang = hang_trong_ngay(goi.test, ngay)
    mu = np.maximum(0, goi.mo_hinh.predict(hang[du_bao.COT_DAC_TRUNG]))
    kich_ban = du_bao.tao_kich_ban_nhu_cau(mu, goi.phan_du, cau_hinh.SO_KICH_BAN,
                                           np.random.default_rng(cau_hinh.HAT_NGAU_NHIEN))
    cac_dong, lich_su_ga = [], None
    ti_le_giua = cac_ti_le[len(cac_ti_le) // 2]

    for ti_le in cac_ti_le:
        bai_toan = tao_bai_toan(goi.thong_so, kich_ban, goi.nhu_cau_tb, ti_le, ti_le)
        g_dp, giay_dp = _do_thoi_gian(lambda: quy_hoach_dong(bai_toan))
        chi_phi_dp = bai_toan.chi_phi(g_dp)

        def them_dong(nhom, ten, chi_phi, do_lech, giay):
            cac_dong.append({"Tỉ lệ ràng buộc": ti_le, "Phương pháp": ten, "Chi phí kỳ vọng": chi_phi,
                             "± Độ lệch chuẩn": do_lech, "Hơn DP (%)": 100 * (chi_phi - chi_phi_dp) / chi_phi_dp,
                             "Thời gian (giây)": giay, "nhom": nhom})

        g, giay = _do_thoi_gian(lambda: phuong_an_thong_thuong(bai_toan, hang["lag_7"].to_numpy()))
        them_dong("thong_thuong", TEN_THONG_THUONG, bai_toan.chi_phi(g), 0.0, giay)
        for cach in CAC_CACH_TINH:
            g, giay = _do_thoi_gian(lambda: tham_lam(bai_toan, cach))
            them_dong("tham_lam", f"{TEN_THAM_LAM} ({TEN_CACH_THAM_LAM[cach]})", bai_toan.chi_phi(g), 0.0, giay)
        for cach_rb, nhom, ten in (("sua", "ga", f"{TEN_GA} – sửa nghiệm"), ("phat", "ga_phat", f"{TEN_GA} – phạt")):
            chi_phi_cac_hat, giay_cac_hat, lich_su = [], [], []
            for hat in range(so_hat):
                (g, ls), giay = _do_thoi_gian(lambda: thuat_toan_di_truyen(bai_toan, cach_rang_buoc=cach_rb, hat=hat))
                chi_phi_cac_hat.append(bai_toan.chi_phi(g))
                giay_cac_hat.append(giay)
                lich_su.append(ls)
            them_dong(nhom, ten, float(np.mean(chi_phi_cac_hat)), float(np.std(chi_phi_cac_hat)), float(np.mean(giay_cac_hat)))
            if cach_rb == "sua" and ti_le == ti_le_giua:
                lich_su_ga = {"ti_le": ti_le, "lich_su": np.array(lich_su), "chi_phi_dp": chi_phi_dp}
        them_dong("dp", TEN_DP, chi_phi_dp, 0.0, giay_dp)

    bang = pd.DataFrame(cac_dong)
    if lich_su_ga is not None:  # các mốc ngang để vẽ cùng đường hội tụ
        da_chon = bang[bang["Tỉ lệ ràng buộc"] == lich_su_ga["ti_le"]]
        lich_su_ga["chi_phi_tham_lam"] = float(da_chon[da_chon["nhom"] == "tham_lam"]["Chi phí kỳ vọng"].min())
        lich_su_ga["chi_phi_thong_thuong"] = float(da_chon[da_chon["nhom"] == "thong_thuong"]["Chi phí kỳ vọng"].iloc[0])
    return bang, lich_su_ga, ngay


# ====================================================================== E3: BACKTEST
def thuc_nghiem_backtest(goi, so_ngay_toi_da=40, ti_le=cau_hinh.TI_LE_NGAN_SACH):
    """Mỗi ngày của tập test: lập kế hoạch bằng dự báo, rồi áp NHU CẦU THẬT để tính chi phí thực tế.

    Ngân sách B và kho V giữ nguyên mọi ngày (đặt từ nhu cầu trung bình của tập train).
    "Dự báo hoàn hảo" biết trước nhu cầu thật: cho thấy dự báo tốt hơn có thể tiết kiệm tối đa bao nhiêu.
    """
    cac_ngay = np.sort(goi.test["ngay"].unique())[:so_ngay_toi_da]
    cac_dong = []
    for stt, ngay in enumerate(cac_ngay):
        hang = hang_trong_ngay(goi.test, ngay)
        mu = np.maximum(0, goi.mo_hinh.predict(hang[du_bao.COT_DAC_TRUNG]))
        nhu_cau_that = hang["so_luong"].to_numpy(float)
        kich_ban = du_bao.tao_kich_ban_nhu_cau(mu, goi.phan_du, cau_hinh.SO_KICH_BAN,
                                               np.random.default_rng(cau_hinh.HAT_NGAU_NHIEN + stt))
        bai_toan = tao_bai_toan(goi.thong_so, kich_ban, goi.nhu_cau_tb, ti_le, ti_le)
        cac_phuong_an = {
            TEN_THONG_THUONG: phuong_an_thong_thuong(bai_toan, hang["lag_7"].to_numpy()),
            TEN_THAM_LAM: tham_lam(bai_toan),
            TEN_GA: thuat_toan_di_truyen(bai_toan, hat=stt)[0],
            TEN_DP: quy_hoach_dong(bai_toan),
        }
        # Dự báo hoàn hảo: chỉ một kịch bản duy nhất = nhu cầu thật (cùng tham số, cùng ràng buộc)
        bai_toan_hoan_hao = tao_bai_toan(goi.thong_so, nhu_cau_that[None, :], goi.nhu_cau_tb, ti_le, ti_le)
        cac_phuong_an[TEN_HOAN_HAO] = quy_hoach_dong(bai_toan_hoan_hao)
        for ten, g in cac_phuong_an.items():
            cac_dong.append({"ngay": ngay, "Phương pháp": ten, **bai_toan.chi_phi_thuc_te(g, nhu_cau_that)})

    theo_ngay = pd.DataFrame(cac_dong)
    chi_phi_cb = theo_ngay[theo_ngay["Phương pháp"] == TEN_THONG_THUONG].set_index("ngay")["tong_chi_phi"]
    theo_ngay["re_hon_thong_thuong"] = theo_ngay["tong_chi_phi"] < theo_ngay["ngay"].map(chi_phi_cb)
    tom_tat = theo_ngay.groupby("Phương pháp", sort=False).agg(
        chi_phi_tb=("tong_chi_phi", "mean"), chi_phi_nhap=("chi_phi_nhap", "mean"),
        chi_phi_ton_kho=("chi_phi_ton_kho", "mean"), chi_phi_thieu=("chi_phi_thieu", "mean"),
        ti_le_dap_ung=("ti_le_dap_ung", "mean"), so_mat_hang_thieu=("so_mat_hang_thieu", "mean"),
        ti_le_ngay_re_hon=("re_hon_thong_thuong", "mean")).reset_index()
    co_so = float(tom_tat.loc[tom_tat["Phương pháp"] == TEN_THONG_THUONG, "chi_phi_tb"].iloc[0])
    tom_tat["tiet_kiem_so_voi_thong_thuong_%"] = 100 * (co_so - tom_tat["chi_phi_tb"]) / co_so
    tom_tat["ti_le_dap_ung"] *= 100
    tom_tat["ti_le_ngay_re_hon"] *= 100
    tom_tat = tom_tat.rename(columns={
        "chi_phi_tb": "Chi phí TB/ngày", "chi_phi_nhap": "– nhập hàng", "chi_phi_ton_kho": "– tồn kho",
        "chi_phi_thieu": "– thiếu hàng", "ti_le_dap_ung": "Tỉ lệ đáp ứng (%)",
        "so_mat_hang_thieu": "Số mặt hàng thiếu/ngày", "ti_le_ngay_re_hon": "% ngày rẻ hơn thông thường",
        "tiet_kiem_so_voi_thong_thuong_%": "Tiết kiệm so với thông thường (%)"})
    return theo_ngay, tom_tat
