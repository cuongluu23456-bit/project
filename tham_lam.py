import numpy as np

from bai_toan import BaiToan

CAC_CACH_TINH = (
    "tong_chuan_hoa",
    "ngan_sach",
    "tuyet_doi",
)


def tham_lam(
    bai_toan: BaiToan,
    cach: str = "tong_chuan_hoa",
) -> np.ndarray:
    """Tìm phương án nhập hàng bằng thuật toán tham lam."""

    if cach not in CAC_CACH_TINH:
        raise ValueError(f"cach tinh khong hop le: {cach}")

    n = bai_toan.n
    bang_chi_phi = bai_toan.bang_chi_phi
    so_thung_toi_da = bai_toan.so_thung_toi_da
    tien_moi_thung = bai_toan.tien_moi_thung
    kho_moi_thung = bai_toan.kho_moi_thung
    ngan_sach = bai_toan.ngan_sach
    suc_chua = bai_toan.suc_chua

    # khởi tạo mảng mà mỗi phần tử là số thùng cần nhập cho một mặt hàng
    so_thung = np.zeros(n, dtype=int)

    # tính mẫu số
    if cach == "tuyet_doi":
        mau_so = np.ones(n)

    elif cach == "ngan_sach":
        mau_so = tien_moi_thung

    else:
        mau_so = tien_moi_thung / ngan_sach + kho_moi_thung / suc_chua

    tien_da_dung = 0.0
    kho_da_dung = 0.0

    while True:
        tot_nhat = -1
        diem_max = 0.0

        for i in range(n):
            g = so_thung[i]

            # giới hạn số thùng
            if g >= so_thung_toi_da[i]:
                continue

            # khả năng nhập thêm một thùng
            if tien_da_dung + tien_moi_thung[i] > ngan_sach * (1 + 1e-9):
                continue

            if kho_da_dung + kho_moi_thung[i] > suc_chua * (1 + 1e-9):
                continue

            # mức giảm chi phí kỳ vọng khi nhập thêm một thùng.
            loi_ich = bang_chi_phi[i, g] - bang_chi_phi[i, g + 1]

            if loi_ich <= 0:
                continue

            # điểm càng lớn, lựa chọn càng được ưu tiên.
            diem = loi_ich / mau_so[i] if mau_so[i] > 0 else np.inf

            if diem > diem_max:
                diem_max = diem
                tot_nhat = i

        # không còn lựa chọn khả thi giúp giảm chi phí.
        if tot_nhat == -1:
            break

        # thực hiện lựa chọn tốt nhất.
        so_thung[tot_nhat] += 1
        tien_da_dung += tien_moi_thung[tot_nhat]
        kho_da_dung += kho_moi_thung[tot_nhat]

    return so_thung
