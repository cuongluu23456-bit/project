"""Dữ liệu bán hàng theo ngày: đọc từ CSV, hoặc sinh dữ liệu GIẢ để chạy thử.

Định dạng chuẩn (mỗi dòng = một mặt hàng trong một ngày cửa hàng mở cửa):
    ngay, ma_hang, ten_hang, so_luong, don_gia

Dùng dữ liệu giả khi chưa có file thật: cả nhóm có thể làm việc song song ngay từ ngày đầu.
Chạy `python du_lieu.py` để ghi một file CSV giả vào data/ban_hang_gia.csv xem thử.
"""
import numpy as np
import pandas as pd

import cau_hinh

CAC_COT_BAT_BUOC = ["ngay", "ma_hang", "ten_hang", "so_luong", "don_gia"]


def sinh_du_lieu_gia(so_mat_hang=cau_hinh.SO_MAT_HANG, so_ngay=330, hat=cau_hinh.HAT_NGAU_NHIEN):
    """Sinh dữ liệu GIẢ có quy luật rõ ràng: thứ trong tuần, xu hướng chậm, khuyến mãi.

    Khuyến mãi làm giá giảm 15% và nhu cầu tăng gấp 1.8–3 lần, nên đặc trưng "giá" có ích cho dự báo.
    Cửa hàng đóng cửa thứ Bảy (dữ liệu UCI cũng không có giao dịch thứ Bảy).
    """
    rng = np.random.default_rng(hat)
    ngay = pd.date_range("2011-01-03", periods=so_ngay, freq="D")  # 03/01/2011 là thứ Hai
    ngay = ngay[ngay.weekday != 5]
    thu = ngay.weekday.to_numpy()
    he_so_thu_chung = np.array([0.9, 1.0, 1.1, 1.15, 1.3, 0.0, 0.7])  # thứ 2 ... Chủ nhật
    cac_bang = []
    for i in range(so_mat_hang):
        nhu_cau_nen = float(np.clip(rng.lognormal(2.6, 0.7), 2.0, 80.0))  # số đơn vị bán trung bình mỗi ngày
        gia_goc = float(np.clip(rng.lognormal(1.3, 0.5), 1.0, 15.0))
        lech_thu = 1 + rng.normal(0, 0.15, 7)                  # mỗi mặt hàng lệch nhẹ so với mẫu chung
        pha = rng.uniform(0, 2 * np.pi)
        xu_huong = 1 + 0.15 * np.sin(2 * np.pi * np.arange(len(ngay)) / 110 + pha)
        khuyen_mai = rng.random(len(ngay)) < 0.04
        he_so = (he_so_thu_chung[thu] * lech_thu[thu] * xu_huong
                 * np.where(khuyen_mai, rng.uniform(1.8, 3.0, len(ngay)), 1.0))
        cuong_do = nhu_cau_nen * he_so * rng.gamma(6.0, 1 / 6.0, len(ngay))  # nhiễu gamma -> số bán dao động mạnh hơn Poisson
        so_luong = rng.poisson(cuong_do)
        don_gia = np.round(gia_goc * (1 + rng.normal(0, 0.02, len(ngay))) * np.where(khuyen_mai, 0.85, 1.0), 2)
        cac_bang.append(pd.DataFrame({
            "ngay": ngay, "ma_hang": f"SP{i + 1:03d}", "ten_hang": f"Mặt hàng giả {i + 1:02d}",
            "so_luong": so_luong, "don_gia": don_gia,
        }))
    return pd.concat(cac_bang, ignore_index=True).sort_values(["ngay", "ma_hang"]).reset_index(drop=True)


def doc_du_lieu(duong_dan):
    """Đọc CSV định dạng chuẩn, kiểm tra cột và sắp xếp theo (ngày, mã hàng)."""
    du_lieu = pd.read_csv(duong_dan, parse_dates=["ngay"])
    thieu = [cot for cot in CAC_COT_BAT_BUOC if cot not in du_lieu.columns]
    if thieu:
        raise ValueError(f"File {duong_dan} thiếu cột: {thieu}. Cần các cột: {CAC_COT_BAT_BUOC}")
    du_lieu["ma_hang"] = du_lieu["ma_hang"].astype(str)
    return du_lieu.sort_values(["ngay", "ma_hang"]).reset_index(drop=True)


def tim_thu_dong_cua(du_lieu):
    """Các thứ trong tuần (0 = thứ Hai ... 6 = Chủ nhật) mà cửa hàng không có giao dịch nào."""
    thu_co_ban = set(du_lieu["ngay"].dt.weekday.unique())
    return sorted(set(range(7)) - thu_co_ban)


def lay_du_lieu(duong_dan=None):
    """Trả về (du_lieu, la_du_lieu_gia). Có file thật thì dùng, không thì sinh dữ liệu giả và CẢNH BÁO."""
    duong_dan = duong_dan or cau_hinh.DUONG_DAN_DU_LIEU
    if duong_dan.exists():
        return doc_du_lieu(duong_dan), False
    print(f"[du_lieu] Không thấy {duong_dan}")
    print("[du_lieu] >>> ĐANG DÙNG DỮ LIỆU GIẢ (chỉ để chạy thử). Số liệu KHÔNG có giá trị báo cáo. <<<")
    print("[du_lieu] Cách có dữ liệu thật: xem README, mục 'Dữ liệu'.")
    return sinh_du_lieu_gia(), True


if __name__ == "__main__":
    duong_dan_ra = cau_hinh.THU_MUC_GOC / "data" / "ban_hang_gia.csv"
    du_lieu_gia = sinh_du_lieu_gia()
    du_lieu_gia.to_csv(duong_dan_ra, index=False)
    print(f"Đã ghi {len(du_lieu_gia):,} dòng dữ liệu GIẢ vào {duong_dan_ra}")
    print(du_lieu_gia.head(8).to_string(index=False))
