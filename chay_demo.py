"""Chạy toàn bộ dự án MỘT LỆNH: dự báo -> tối ưu -> kế hoạch ngày mai -> thực nghiệm E1, E2, E3.

    python chay_demo.py                      # dùng data/ban_hang_theo_ngay.csv nếu có, không thì dữ liệu GIẢ
    python chay_demo.py --du-lieu duong/dan.csv
    python chay_demo.py --nhanh              # bản rút gọn để thử nhanh (ít seed, ít ngày backtest)

Kết quả: in bảng ra màn hình và lưu bảng CSV + biểu đồ PNG vào thư mục ket_qua/.
Đây thay cho "dashboard": cùng nội dung (dự báo, số lượng nhập, chi phí, mặt hàng nguy cơ thiếu,
so sánh với cách nhập thông thường) nhưng chỉ cần matplotlib.
"""
import argparse
from pathlib import Path

import cau_hinh
import thuc_nghiem as tn
import ve_hinh
from du_lieu import lay_du_lieu


def in_tieu_de(chuoi):
    print("\n" + "=" * 78 + f"\n{chuoi}\n" + "=" * 78)


def in_bang(bang, so_le=2):
    print(bang.round(so_le).to_string(index=False))


def main():
    boc = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    boc.add_argument("--du-lieu", type=Path, default=None, help="đường dẫn file CSV bán hàng theo ngày")
    boc.add_argument("--nhanh", action="store_true", help="chạy rút gọn để thử nhanh")
    boc.add_argument("--so-hat-ga", type=int, default=None, help="số lần chạy GA (seed) trong E2")
    boc.add_argument("--so-ngay-backtest", type=int, default=None, help="số ngày backtest trong E3")
    boc.add_argument("--ket-qua", type=Path, default=None, help=f"thư mục lưu kết quả (mặc định: {cau_hinh.THU_MUC_KET_QUA.name}/)")
    tham_so = boc.parse_args()
    so_hat = tham_so.so_hat_ga or (5 if tham_so.nhanh else 30)
    so_ngay = tham_so.so_ngay_backtest or (10 if tham_so.nhanh else 40)

    ra = tham_so.ket_qua or cau_hinh.THU_MUC_KET_QUA
    ra.mkdir(parents=True, exist_ok=True)

    du_lieu, la_gia = lay_du_lieu(tham_so.du_lieu)
    in_tieu_de("DỮ LIỆU")
    print(f"{du_lieu['ma_hang'].nunique()} mặt hàng, {du_lieu['ngay'].nunique()} ngày mở cửa "
          f"({du_lieu['ngay'].min():%d/%m/%Y} – {du_lieu['ngay'].max():%d/%m/%Y})"
          + ("  [DỮ LIỆU GIẢ]" if la_gia else ""))
    goi = tn.chuan_bi_chung(du_lieu)

    # ---------------------------------------------------------------- E1
    in_tieu_de("E1 – DỰ BÁO NHU CẦU (sai số càng nhỏ càng tốt)")
    in_bang(goi.bang_du_bao)
    print(f"-> Mô hình ML tốt nhất theo tập valid: {goi.ten_mo_hinh} (dùng cho các bước sau)")
    goi.bang_du_bao.to_csv(ra / "bang_E1_du_bao.csv", index=False)
    ve_hinh.ve_du_bao(goi, ra / "hinh_1_du_bao.png")

    # ---------------------------------------------------------------- Kế hoạch ngày mai
    kh = tn.lap_ke_hoach_ngay_mai(goi)
    bt = kh["bai_toan"]
    in_tieu_de(f"KẾ HOẠCH NHẬP HÀNG NGÀY MAI ({kh['ngay']:%d/%m/%Y})")
    print(f"Ngân sách B = {bt.ngan_sach:,.0f} | Sức chứa kho V = {bt.suc_chua:,.0f} | {cau_hinh.SO_KICH_BAN} kịch bản nhu cầu\n")
    in_bang(kh["bang"], 1)
    print("\nSo sánh các phương án (chi phí KỲ VỌNG, thấp hơn = tốt hơn):")
    in_bang(kh["tong_ket"], 1)
    thieu = kh["bang"][kh["bang"]["Cảnh báo thiếu hàng"] == "CÓ"]
    print(f"\nPhương án đề xuất: {kh['de_xuat']}")
    if len(thieu):
        print("Mặt hàng có nguy cơ thiếu hàng (xác suất thiếu >= "
              f"{int(100 * cau_hinh.NGUONG_NGUY_CO_THIEU)}%): " + ", ".join(thieu["Mã hàng"]))
    else:
        print("Không có mặt hàng nào có nguy cơ thiếu hàng cao.")
    kh["bang"].to_csv(ra / "ke_hoach_ngay_mai.csv", index=False)
    kh["tong_ket"].to_csv(ra / "ke_hoach_ngay_mai_so_sanh.csv", index=False)
    ve_hinh.ve_ke_hoach_ngay_mai(kh, ra / "hinh_2_ke_hoach_ngay_mai.png")

    # ---------------------------------------------------------------- E2
    in_tieu_de(f"E2 – TỐI ƯU CHO MỘT NGÀY, NHIỀU MỨC RÀNG BUỘC (GA chạy {so_hat} seed)")
    bang_e2, lich_su_ga, ngay_e2 = tn.thuc_nghiem_toi_uu_mot_ngay(goi, so_hat=so_hat)
    print(f"Ngày thử: {ngay_e2:%d/%m/%Y}. 'Hơn DP (%)': chi phí cao hơn DP bao nhiêu %; số âm = tốt hơn DP\n"
          "(DP chỉ chính xác trên lưới rời rạc, nên GA/Tham lam đôi khi nhỉnh hơn một chút).\n")
    in_bang(bang_e2.drop(columns="nhom"), 2)
    bang_e2.drop(columns="nhom").to_csv(ra / "bang_E2_toi_uu_mot_ngay.csv", index=False)
    ve_hinh.ve_chi_phi_e2(bang_e2, ra / "hinh_3_E2_chi_phi.png")
    ve_hinh.ve_hoi_tu_ga(lich_su_ga, ra / "hinh_4_hoi_tu_ga.png")

    # ---------------------------------------------------------------- E3
    in_tieu_de(f"E3 – BACKTEST {so_ngay} NGÀY (áp nhu cầu THẬT lên kế hoạch)")
    theo_ngay, tom_tat = tn.thuc_nghiem_backtest(goi, so_ngay_toi_da=so_ngay)
    in_bang(tom_tat, 1)
    theo_ngay.to_csv(ra / "bang_E3_backtest_theo_ngay.csv", index=False)
    tom_tat.to_csv(ra / "bang_E3_backtest_tom_tat.csv", index=False)
    ve_hinh.ve_backtest(tom_tat, ra / "hinh_5_backtest.png")

    in_tieu_de("XONG")
    print(f"Đã lưu {len(list(ra.glob('*.csv')))} bảng CSV và {len(list(ra.glob('*.png')))} biểu đồ vào: {ra}")
    if la_gia:
        print("\n!!! ĐÂY LÀ DỮ LIỆU GIẢ: số liệu chỉ để kiểm tra chương trình, KHÔNG dùng làm kết quả báo cáo. !!!")
        print("    Cách dùng dữ liệu thật: xem README, mục 'Dữ liệu'.")


if __name__ == "__main__":
    main()
