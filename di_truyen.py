import numpy as np

import cau_hinh
from bai_toan import BaiToan


def di_truyen(
    bai_toan: BaiToan,
    kich_thuoc_quan_the: int = cau_hinh.GA_KICH_THUOC_QUAN_THE,
    so_the_he: int = cau_hinh.GA_SO_THE_HE,
    cach_rang_buoc: str = cau_hinh.GA_CACH_RANG_BUOC,
    hat: int = cau_hinh.HAT_NGAU_NHIEN,
) -> tuple[np.ndarray, list[float]]:
    """Trả về (số thùng tốt nhất, lịch sử chi phí tốt nhất qua các thế hệ).

    ``cach_rang_buoc='sua'``: sửa cá thể vi phạm bằng cách co theo tỉ lệ.
    ``cach_rang_buoc='phat'``: giữ cá thể vi phạm nhưng cộng phạt khi chọn lọc.
    Nghiệm trả về luôn hợp lệ; thuật toán không bảo đảm tối ưu toàn cục.
    """
    if cach_rang_buoc not in ("sua", "phat"):
        raise ValueError("cach_rang_buoc phai la 'sua' hoac 'phat'")
    if cach_rang_buoc == "phat":
        raise ValueError("cach_rang_buoc phat chua trien khai")
    if so_the_he < 1:
        raise ValueError("so_the_he phai >= 1")

    so_uu_tu = cau_hinh.GA_SO_UU_TU
    kich_thuoc_giai_dau = cau_hinh.GA_KICH_THUOC_GIAI_DAU
    xac_suat_lai = cau_hinh.GA_XAC_SUAT_LAI
    if not 1 <= so_uu_tu < kich_thuoc_quan_the:
        raise ValueError("so_uu_tu khong the > kich_thuoc_quan_the")
    if kich_thuoc_giai_dau < 1 or not 0 <= xac_suat_lai <= 1:
        raise ValueError("kich_thuoc_giai_dau hoac xac_suat_lai khong hop le")

    n = bai_toan.n
    if n < 1:
        raise ValueError("bai_toan.n khong < 1")

    rng = np.random.default_rng(hat)
    bang_chi_phi = bai_toan.bang_chi_phi
    so_thung_toi_da = bai_toan.so_thung_toi_da
    tien_moi_thung = bai_toan.tien_moi_thung
    kho_moi_thung = bai_toan.kho_moi_thung
    ngan_sach = bai_toan.ngan_sach
    suc_chua = bai_toan.suc_chua

    # khởi tạo ngẫu nhiên
    quan_the = rng.integers(0, so_thung_toi_da + 1, size=(kich_thuoc_quan_the, n))
    quan_the = bai_toan.sua_nghiem_co_ty_le(quan_the)

    # tạo nghiệm không nhập hàng
    quan_the[0, : ] = 0

    # tạo một nghiệm có chi phí tốt
    muc_tot_rieng = np.array([
        np.argmin(bang_chi_phi[i, :so_thung_toi_da[i] + 1])
        for i in range(n)
    ], dtype=int)
    quan_the[1] = bai_toan.sua_nghiem_tham_lam(muc_tot_rieng)

    tot_nhat = quan_the[0].copy()
    chi_phi_tot_nhat = bai_toan.chi_phi(tot_nhat)
    lich_su = []
    so_con = kich_thuoc_quan_the - so_uu_tu
    so_cap = (so_con + 1) // 2

    for the_he in range(so_the_he):
        # tính chi phí của toàn bộ quần thể
        chi_phi = bai_toan.chi_phi_quan_the(quan_the)
        tien_dung = quan_the @ tien_moi_thung
        kho_dung = quan_the @ kho_moi_thung
        hop_le = ((tien_dung <= ngan_sach * (1 + 1e-9))
                  & (kho_dung <= suc_chua * (1 + 1e-9)))

        # ghi nhớ nghiệm hợp lệ tốt nhất đã tìm được qua tất cả các thế hệ
        vi_tri_hop_le = np.flatnonzero(hop_le)
        if len(vi_tri_hop_le):
            i = vi_tri_hop_le[np.argmin(chi_phi[vi_tri_hop_le])]
            if chi_phi[i] < chi_phi_tot_nhat:
                chi_phi_tot_nhat = chi_phi[i]
                tot_nhat = quan_the[i].copy()
        lich_su.append(chi_phi_tot_nhat)

        if the_he == so_the_he - 1:
            break

        # giữ lại những cá thể tốt nhất
        thu_tu = np.argsort(chi_phi)
        quan_the_moi = np.empty_like(quan_the)
        quan_the_moi[:so_uu_tu] = quan_the[thu_tu[:so_uu_tu]]

        # chọn cha mẹ bằng giải đấu
        cac_vi_tri = rng.integers(
            0, kich_thuoc_quan_the,
            size=(so_cap * 2, kich_thuoc_giai_dau),
        )

        vi_tri_tot = np.argmin(chi_phi[cac_vi_tri], axis=1)
        vi_tri_thang = cac_vi_tri[np.arange(so_cap * 2), vi_tri_tot]

        cha = quan_the[vi_tri_thang[:so_cap]]
        me = quan_the[vi_tri_thang[so_cap:]]

        # lai ghép đồng nhất: mỗi gen được lấy từ cha hoặc mẹ với xác suất 1/2
        # nếu không lai ghép (10% mặc định), giữ nguyên cả hai cha mẹ
        co_lai = rng.random(so_cap) < xac_suat_lai
        mat_na_lai = (rng.random((so_cap, n)) < 0.5) & co_lai[:, None]
        con = np.empty((so_cap * 2, n), dtype=int)
        con[0::2] = np.where(mat_na_lai, cha, me)
        con[1::2] = np.where(mat_na_lai, me, cha)
        con = con[:so_con]

        # xác suất đột biến mỗi gen là 1/n (trung bình một gen mỗi cá thể)
        # phần lớn thay đổi 1–2 thùng; đôi khi đặt lại một gen ngẫu nhiên
        # để vượt qua các mức chi phí cố định gây ra cực trị địa phương
        dot_bien = rng.random(con.shape) < 1 / n
        dat_lai = dot_bien & (rng.random(con.shape) < 0.2)
        buoc = rng.integers(1, 3, size=con.shape)
        huong = np.where(rng.random(con.shape) < 0.5, -1, 1)
        con += np.where(dot_bien & ~dat_lai, buoc * huong, 0)
        gia_tri_moi = rng.integers(0, so_thung_toi_da + 1, size=con.shape)
        con[dat_lai] = gia_tri_moi[dat_lai]
        np.clip(con, 0, so_thung_toi_da, out=con)

        con = bai_toan.sua_nghiem_co_ty_le(con)

        quan_the_moi[so_uu_tu:] = con
        quan_the = quan_the_moi

    return tot_nhat, lich_su
