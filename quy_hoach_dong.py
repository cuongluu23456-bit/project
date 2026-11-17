import numpy as np

import cau_hinh
from bai_toan import BaiToan


def quy_hoach_dong(
    bai_toan: BaiToan,
    so_o_ngan_sach: int = cau_hinh.LUOI_DP_NGAN_SACH,
    so_o_kho: int = cau_hinh.LUOI_DP_KHO,
) -> np.ndarray:
    """Tìm phương án nhập hàng có chi phí nhỏ nhất trên lưới DP."""

    B, V = so_o_ngan_sach, so_o_kho
    bang_chi_phi = bai_toan.bang_chi_phi

    # quy đổi tài nguyên mỗi thùng sang số ô
    tien_moi_thung = bai_toan.tien_moi_thung * B / bai_toan.ngan_sach
    kho_moi_thung = bai_toan.kho_moi_thung * V / bai_toan.suc_chua

    # bảng DP và thông tin truy vết
    previous = np.zeros((B + 1, V + 1))
    da_chon = np.zeros((bai_toan.n, B + 1, V + 1), dtype=np.int32)

    cac_o_tien = []
    cac_o_kho = []

    for i in range(bai_toan.n):
        g = np.arange(bai_toan.so_thung_toi_da[i] + 1)

        o_tien = np.ceil(g * tien_moi_thung[i]).astype(int)
        o_kho = np.ceil(g * kho_moi_thung[i]).astype(int)

        cac_o_tien.append(o_tien)
        cac_o_kho.append(o_kho)

        # g = 0: không nhập mặt hàng hiện tại
        current = previous + bang_chi_phi[i, 0]

        for so_thung in range(1, len(g)):
            t = int(o_tien[so_thung])
            k = int(o_kho[so_thung])

            if t > B or k > V:
                break

            # công thức chuyển trạng thái DP
            ung_vien = (
                previous[:B + 1 - t, :V + 1 - k]
                + bang_chi_phi[i, so_thung]
            )

            vung_hien_tai = current[t:, k:]
            tot_hon = ung_vien < vung_hien_tai

            vung_hien_tai[tot_hon] = ung_vien[tot_hon]
            da_chon[i, t:, k:][tot_hon] = so_thung

        previous = current

    # truy vết số thùng đã chọn
    so_thung = np.zeros(bai_toan.n, dtype=int)
    b, v = B, V

    for i in range(bai_toan.n - 1, -1, -1):
        g = int(da_chon[i, b, v])

        so_thung[i] = g
        b -= int(cac_o_tien[i][g])
        v -= int(cac_o_kho[i][g])

    return so_thung
