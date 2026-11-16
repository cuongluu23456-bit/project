"""PHẦN 1 – Machine Learning: dự báo nhu cầu của từng mặt hàng cho một ngày.

Ý tưởng:
  * Một mô hình chung cho cả 20 mặt hàng, dự báo số lượng bán của ngày t.
  * Đặc trưng chỉ dùng thông tin biết TRƯỚC ngày t: thứ trong tuần, giá, doanh số hôm qua,
    doanh số cùng thứ tuần trước, trung bình 7 ngày gần nhất. TUYỆT ĐỐI không dùng doanh số của chính ngày t.
  * Chia dữ liệu theo THỜI GIAN (train -> valid -> test), không chia ngẫu nhiên.
  * So sánh với 2 phương pháp cơ sở đơn giản; mô hình ML phải thắng chúng mới có ý nghĩa.
  * Từ sai số trên tập valid, sinh nhiều "kịch bản nhu cầu" để phần tối ưu tính được rủi ro thừa/thiếu hàng.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import cau_hinh
from du_lieu import tim_thu_dong_cua

COT_DAC_TRUNG = ["ma_hang_so", "thu", "don_gia", "lag_1", "lag_7", "tb_7"]
TEN_TUAN_TRUOC = "Bằng tuần trước (cơ sở)"
TEN_TRUNG_BINH_7 = "Trung bình 7 ngày (cơ sở)"


# ------------------------------------------------------------------ đặc trưng
def tao_dac_trung(du_lieu):
    """Tạo bảng đặc trưng. Mỗi dòng = (ngày, mặt hàng). Cột đích: so_luong.

    lag_1  : doanh số của lần mở cửa trước đó
    lag_7  : doanh số cùng thứ tuần trước (theo lịch; nếu thiếu thì dùng tb_7)
    tb_7   : trung bình doanh số 7 lần mở cửa gần nhất (không gồm ngày hiện tại)
    """
    d = du_lieu.sort_values(["ma_hang", "ngay"]).reset_index(drop=True).copy()
    d["ma_hang_so"] = d["ma_hang"].astype("category").cat.codes
    d["thu"] = d["ngay"].dt.weekday
    theo_hang = d.groupby("ma_hang")["so_luong"]
    d["lag_1"] = theo_hang.shift(1)  # shift(1): lùi 1 dòng -> chỉ thấy quá khứ
    d["tb_7"] = theo_hang.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    tuan_truoc = d[["ngay", "ma_hang", "so_luong"]].copy()
    tuan_truoc["ngay"] = tuan_truoc["ngay"] + pd.Timedelta(days=7)  # doanh số ngày X xuất hiện ở ngày X+7
    tuan_truoc = tuan_truoc.rename(columns={"so_luong": "lag_7"})
    d = d.merge(tuan_truoc, on=["ngay", "ma_hang"], how="left")
    d["lag_7"] = d["lag_7"].fillna(d["tb_7"])
    return d.dropna(subset=["tb_7"]).reset_index(drop=True)  # bỏ ngày đầu tiên của mỗi mặt hàng (chưa có quá khứ)


def chia_theo_thoi_gian(dac_trung):
    """Chia theo ngày: train (đầu) -> valid -> test (cuối). Trả về 3 bảng."""
    cac_ngay = np.sort(dac_trung["ngay"].unique())
    cuoi_train = int(len(cac_ngay) * cau_hinh.TI_LE_TRAIN)
    cuoi_valid = int(len(cac_ngay) * (cau_hinh.TI_LE_TRAIN + cau_hinh.TI_LE_VALID))
    la_train = dac_trung["ngay"].isin(cac_ngay[:cuoi_train])
    la_valid = dac_trung["ngay"].isin(cac_ngay[cuoi_train:cuoi_valid])
    la_test = dac_trung["ngay"].isin(cac_ngay[cuoi_valid:])
    return dac_trung[la_train], dac_trung[la_valid], dac_trung[la_test]


# ------------------------------------------------------------------ mô hình và đánh giá
def tao_cac_mo_hinh(hat=cau_hinh.HAT_NGAU_NHIEN):
    return {
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=hat),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=hat),
    }


def danh_gia(y_that, y_du_bao):
    """MAE: sai số tuyệt đối trung bình (đơn vị hàng). WAPE: tổng sai số / tổng doanh số (%)."""
    sai_so = np.abs(np.asarray(y_that, float) - np.asarray(y_du_bao, float))
    return {"MAE": float(sai_so.mean()), "WAPE (%)": float(100 * sai_so.sum() / max(np.sum(y_that), 1e-9))}


def so_sanh_cac_mo_hinh(train, valid, test):
    """Huấn luyện 2 mô hình ML, so với 2 cơ sở. Trả về (bảng kết quả, các mô hình, tên mô hình tốt nhất)."""
    cac_mo_hinh = tao_cac_mo_hinh()
    for mo_hinh in cac_mo_hinh.values():
        mo_hinh.fit(train[COT_DAC_TRUNG], train["so_luong"])

    cach_du_bao = {
        TEN_TUAN_TRUOC: lambda bang: bang["lag_7"].to_numpy(),
        TEN_TRUNG_BINH_7: lambda bang: bang["tb_7"].to_numpy(),
    }
    for ten, mo_hinh in cac_mo_hinh.items():
        cach_du_bao[ten] = lambda bang, mh=mo_hinh: mh.predict(bang[COT_DAC_TRUNG])

    cac_dong = []
    for ten, ham in cach_du_bao.items():
        kq_valid = danh_gia(valid["so_luong"], ham(valid))
        kq_test = danh_gia(test["so_luong"], ham(test))
        cac_dong.append({"Phương pháp": ten, "MAE (valid)": kq_valid["MAE"], "WAPE % (valid)": kq_valid["WAPE (%)"],
                         "MAE (test)": kq_test["MAE"], "WAPE % (test)": kq_test["WAPE (%)"]})
    bang_ket_qua = pd.DataFrame(cac_dong)
    mae_valid = bang_ket_qua.set_index("Phương pháp")["MAE (valid)"]
    ten_tot_nhat = min(cac_mo_hinh, key=lambda ten: mae_valid[ten])  # chọn theo valid, không nhìn test
    return bang_ket_qua, cac_mo_hinh, ten_tot_nhat


# ------------------------------------------------------------------ kịch bản nhu cầu (bất định)
def tinh_phan_du(mo_hinh, valid):
    """Sai số dự báo (thực tế - dự báo) trên tập valid, tách theo từng mặt hàng (thứ tự ma_hang_so)."""
    phan_du = []
    for _, bang in valid.groupby("ma_hang_so", sort=True):
        du_bao = mo_hinh.predict(bang[COT_DAC_TRUNG])
        phan_du.append(bang["so_luong"].to_numpy() - du_bao)
    return phan_du


def tao_kich_ban_nhu_cau(mu, phan_du, so_kich_ban, rng):
    """Sinh S kịch bản nhu cầu: nhu cầu = dự báo + sai số lấy mẫu lại (bootstrap) từ sai số quá khứ.

    Dự báo chỉ là MỘT con số; thực tế nhu cầu dao động quanh nó. Các kịch bản mô tả sự dao động đó,
    nhờ vậy chi phí tồn kho và thiếu hàng mới có ý nghĩa. Trả về mảng (so_kich_ban, so_mat_hang).
    """
    cac_cot = []
    for i, du_bao in enumerate(mu):
        sai_so = rng.choice(phan_du[i], size=so_kich_ban, replace=True)
        cac_cot.append(np.maximum(0, np.rint(du_bao + sai_so)))
    return np.column_stack(cac_cot)


# ------------------------------------------------------------------ dự báo ngày mai
def ngay_mo_cua_ke_tiep(ngay_cuoi, thu_dong_cua):
    ngay = ngay_cuoi + pd.Timedelta(days=1)
    while ngay.weekday() in thu_dong_cua:
        ngay += pd.Timedelta(days=1)
    return ngay


def du_bao_ngay_mai(du_lieu, mo_hinh):
    """Dự báo nhu cầu cho ngày mở cửa kế tiếp. Trả về (ngày, bảng đặc trưng theo mặt hàng, dự báo mu).

    Ghép thêm một dòng "ngày mai" (chưa có doanh số) rồi dùng đúng hàm tạo đặc trưng như khi huấn luyện;
    các đặc trưng lag chỉ lấy từ quá khứ nên không có rò rỉ.
    """
    ngay_mai = ngay_mo_cua_ke_tiep(du_lieu["ngay"].max(), tim_thu_dong_cua(du_lieu))
    dong_cuoi = du_lieu.sort_values("ngay").groupby("ma_hang").tail(1)
    dong_mai = pd.DataFrame({"ngay": ngay_mai, "ma_hang": dong_cuoi["ma_hang"].to_numpy(),
                             "ten_hang": dong_cuoi["ten_hang"].to_numpy(), "so_luong": np.nan,
                             "don_gia": dong_cuoi["don_gia"].to_numpy()})
    dac_trung = tao_dac_trung(pd.concat([du_lieu, dong_mai], ignore_index=True))
    hang_mai = dac_trung[dac_trung["ngay"] == ngay_mai].sort_values("ma_hang_so").reset_index(drop=True)
    mu = np.maximum(0, mo_hinh.predict(hang_mai[COT_DAC_TRUNG]))
    return ngay_mai, hang_mai, mu
