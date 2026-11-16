"""Chạy MỘT LẦN bởi MỘT người: biến file UCI Online Retail thành CSV nhỏ (20 mặt hàng × ngày mở cửa).

Cách dùng:
    python chuan_bi_du_lieu_uci.py "Online Retail.xlsx"
    python chuan_bi_du_lieu_uci.py online_retail.csv --so-mat-hang 20

Tải dữ liệu: https://archive.ics.uci.edu/dataset/352/online-retail  (file gốc thường là Excel .xlsx,
đọc file Excel cần thư viện openpyxl: pip install openpyxl). Cả nhóm chỉ cần file CSV đầu ra.

Các bước làm sạch:
  1. Bỏ hóa đơn hủy (mã hóa đơn bắt đầu bằng "C"), số lượng <= 0, giá <= 0.
  2. Chỉ giữ mã sản phẩm thật (bắt đầu bằng 5 chữ số), bỏ các mã như POST, BANK CHARGES...
  3. Gộp theo (ngày, mặt hàng). Ngày không có giao dịch nào của cả cửa hàng = đóng cửa, bỏ khỏi chuỗi.
  4. Chọn N mặt hàng có số ngày bán nhiều nhất (nhu cầu đều đặn, dự báo có ý nghĩa).
  5. Ngày cửa hàng mở nhưng mặt hàng không bán = 0. Cắt ngoại lệ (đơn sỉ rất lớn) ở phân vị 99.
"""
import argparse
from pathlib import Path

import pandas as pd

import cau_hinh


def doc_file_uci(duong_dan):
    """Đọc file UCI dạng .xlsx hoặc .csv."""
    duong_dan = Path(duong_dan)
    if duong_dan.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(duong_dan)
    try:
        return pd.read_csv(duong_dan, encoding="utf-8")
    except UnicodeDecodeError:  # bản CSV trên mạng thường mã hóa latin-1
        return pd.read_csv(duong_dan, encoding="latin-1")


def lam_sach(giao_dich):
    """Bước 1–2: giữ lại các giao dịch bán hàng hợp lệ của sản phẩm thật."""
    g = giao_dich.copy()
    g["StockCode"] = g["StockCode"].astype(str)
    g = g[~g["InvoiceNo"].astype(str).str.startswith("C")]
    g = g[(g["Quantity"] > 0) & (g["UnitPrice"] > 0)]
    g = g[g["StockCode"].str.match(r"^\d{5}")]
    g["ngay"] = pd.to_datetime(g["InvoiceDate"]).dt.normalize()
    return g


def chon_mat_hang(g, so_mat_hang):
    """Bước 4: chọn các mặt hàng bán đều nhất (nhiều ngày có bán nhất, hòa thì xét tổng số lượng)."""
    thong_ke = g.groupby("StockCode").agg(so_ngay_ban=("ngay", "nunique"), tong_so_luong=("Quantity", "sum"))
    thong_ke = thong_ke.sort_values(["so_ngay_ban", "tong_so_luong"], ascending=False)
    return list(thong_ke.head(so_mat_hang).index)


def tao_bang_theo_ngay(g, cac_ma_hang):
    """Bước 3 và 5: bảng đầy đủ (ngày mở cửa × mặt hàng)."""
    ngay_mo_cua = sorted(g["ngay"].unique())
    gc = g[g["StockCode"].isin(cac_ma_hang)]
    theo_ngay = gc.groupby(["ngay", "StockCode"]).agg(so_luong=("Quantity", "sum"), don_gia=("UnitPrice", "mean"))
    luoi = pd.MultiIndex.from_product([ngay_mo_cua, cac_ma_hang], names=["ngay", "StockCode"])
    bang = theo_ngay.reindex(luoi).reset_index()
    bang["so_luong"] = bang["so_luong"].fillna(0)
    # ngày không bán: lấy giá của lần bán gần nhất (hoặc kế tiếp nếu chưa từng bán trước đó)
    bang["don_gia"] = bang.groupby("StockCode")["don_gia"].transform(lambda s: s.ffill().bfill())
    gioi_han = bang.groupby("StockCode")["so_luong"].transform(lambda s: s.quantile(0.99))
    bang["so_luong"] = bang["so_luong"].clip(upper=gioi_han).round().astype(int)
    ten = gc.groupby("StockCode")["Description"].agg(lambda s: s.dropna().astype(str).iloc[0] if s.notna().any() else "")
    bang["ten_hang"] = bang["StockCode"].map(ten)
    bang = bang.rename(columns={"StockCode": "ma_hang"})
    return bang[["ngay", "ma_hang", "ten_hang", "so_luong", "don_gia"]].sort_values(["ngay", "ma_hang"])


def chuan_bi(duong_dan_vao, duong_dan_ra, so_mat_hang):
    giao_dich = doc_file_uci(duong_dan_vao)
    g = lam_sach(giao_dich)
    cac_ma_hang = chon_mat_hang(g, so_mat_hang)
    bang = tao_bang_theo_ngay(g, cac_ma_hang)
    duong_dan_ra.parent.mkdir(parents=True, exist_ok=True)
    bang.to_csv(duong_dan_ra, index=False)
    print(f"Đã ghi {len(bang):,} dòng ({bang['ngay'].nunique()} ngày × {len(cac_ma_hang)} mặt hàng) vào {duong_dan_ra}")
    return bang


def main():
    boc_tham_so = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    boc_tham_so.add_argument("file_uci", help="đường dẫn file Online Retail (.xlsx hoặc .csv)")
    boc_tham_so.add_argument("--so-mat-hang", type=int, default=cau_hinh.SO_MAT_HANG)
    boc_tham_so.add_argument("--ra", type=Path, default=cau_hinh.DUONG_DAN_DU_LIEU)
    tham_so = boc_tham_so.parse_args()
    chuan_bi(tham_so.file_uci, tham_so.ra, tham_so.so_mat_hang)


if __name__ == "__main__":
    main()
