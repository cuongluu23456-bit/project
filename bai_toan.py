"""PHẦN 2 – Bài toán tối ưu nhập hàng cho MỘT ngày.

Quyết định: nhập bao nhiêu THÙNG của mỗi mặt hàng. Vector quyết định `so_thung` gồm n số nguyên:
    số lượng nhập q_i = so_thung_i × quy_cach_i

Hàm mục tiêu: chi phí KỲ VỌNG trên S kịch bản nhu cầu d_is (xem du_bao.tao_kich_ban_nhu_cau)
    Q(q) = Σ_i [ c_i·q_i + K_i·1(q_i>0) ]                    chi phí nhập hàng (+ phí cố định mỗi lần nhập)
         + (1/S)·Σ_s Σ_i [ h_i·max(q_i − d_is, 0)            chi phí tồn kho: mỗi đơn vị dư
                         + P_i·max(d_is − q_i, 0) ]          chi phí thiếu hàng: doanh thu bị mất
Ràng buộc:
    Σ c_i·q_i ≤ B   (ngân sách nhập hàng)
    Σ v_i·q_i ≤ V   (sức chứa kho)

Ký hiệu trong code:  P_i = gia_ban,  c_i = gia_nhap,  v_i = the_tich,  h_i = phi_ton_kho,
                     K_i = phi_co_dinh,  k_i = quy_cach,  d_is = kich_ban,  B = ngan_sach,  V = suc_chua.

Mẹo quan trọng: Q cộng TÁCH theo từng mặt hàng, nên ta tính trước bảng `bang_chi_phi[i][g]`
(chi phí kỳ vọng của mặt hàng i nếu nhập g thùng). Sau đó mọi thuật toán chỉ việc tra bảng, rất nhanh.

Kiểm tra đúng sai: nếu B, V rất lớn, K_i = 0 và quy cách = 1 thì nghiệm tối ưu của mặt hàng i là phân vị mức
(P_i − c_i)/(P_i + h_i) của các kịch bản nhu cầu (công thức "người bán báo" – newsvendor).
"""
import numpy as np

import cau_hinh


class BaiToan:
    """Một bài toán nhập hàng cụ thể: giá, ràng buộc và các kịch bản nhu cầu."""

    def __init__(self, gia_ban, gia_nhap, the_tich, phi_ton_kho, phi_co_dinh, quy_cach,
                 kich_ban, ngan_sach, suc_chua):
        self.gia_ban = np.asarray(gia_ban, float)
        self.gia_nhap = np.asarray(gia_nhap, float)
        self.the_tich = np.asarray(the_tich, float)
        self.phi_ton_kho = np.asarray(phi_ton_kho, float)
        self.phi_co_dinh = np.asarray(phi_co_dinh, float)
        self.quy_cach = np.asarray(quy_cach, int)
        self.kich_ban = np.asarray(kich_ban, float)  # kích thước (S kịch bản, n mặt hàng)
        self.ngan_sach = float(ngan_sach)
        self.suc_chua = float(suc_chua)
        self.n = len(self.gia_ban)

        self.tien_moi_thung = self.gia_nhap * self.quy_cach   # tiền cho mỗi thùng
        self.kho_moi_thung = self.the_tich * self.quy_cach    # thể tích kho cho mỗi thùng
        # Nhập quá nhu cầu lớn nhất trong mọi kịch bản là vô ích -> chặn trên số thùng
        self.so_thung_toi_da = np.ceil(self.kich_ban.max(axis=0) / self.quy_cach).astype(int)
        self.bang_chi_phi = self._lap_bang_chi_phi()

    def _lap_bang_chi_phi(self):
        """bang[i][g] = chi phí kỳ vọng của mặt hàng i khi nhập g thùng (g = 0 ... so_thung_toi_da[i])."""
        so_cot = int(self.so_thung_toi_da.max()) + 1
        bang = np.full((self.n, so_cot), np.inf)
        for i in range(self.n):
            q = self.quy_cach[i] * np.arange(self.so_thung_toi_da[i] + 1)  # các số lượng có thể nhập
            nhu_cau = self.kich_ban[:, i][:, None]                          # (S, 1)
            hang_du = np.maximum(q[None, :] - nhu_cau, 0).mean(axis=0)      # trung bình số đơn vị dư
            hang_thieu = np.maximum(nhu_cau - q[None, :], 0).mean(axis=0)   # trung bình số đơn vị thiếu
            bang[i, :len(q)] = (self.gia_nhap[i] * q + self.phi_co_dinh[i] * (q > 0)
                                + self.phi_ton_kho[i] * hang_du + self.gia_ban[i] * hang_thieu)
        return bang

    # ---------------------------------------------------------------- tính toán trên một phương án
    def so_luong(self, so_thung):
        return np.asarray(so_thung) * self.quy_cach

    def chi_phi(self, so_thung):
        """Chi phí kỳ vọng Q của một phương án (tra bảng)."""
        return float(self.bang_chi_phi[np.arange(self.n), so_thung].sum())

    def chi_phi_quan_the(self, quan_the):
        """Chi phí của cả một quần thể (ma trận P × n) cùng lúc – dùng cho thuật toán di truyền."""
        return self.bang_chi_phi[np.arange(self.n)[None, :], quan_the].sum(axis=1)

    def tien_nhap(self, so_thung):
        return np.asarray(so_thung) @ self.tien_moi_thung

    def the_tich_dung(self, so_thung):
        return np.asarray(so_thung) @ self.kho_moi_thung

    def hop_le(self, so_thung):
        """Phương án có thỏa cả hai ràng buộc không? (cho sai số làm tròn rất nhỏ)"""
        return bool(self.tien_nhap(so_thung) <= self.ngan_sach * (1 + 1e-9)
                    and self.the_tich_dung(so_thung) <= self.suc_chua * (1 + 1e-9))

    def xac_suat_thieu(self, so_thung):
        """Với mỗi mặt hàng: tỉ lệ kịch bản mà nhu cầu lớn hơn số lượng nhập (nguy cơ thiếu hàng)."""
        return (self.kich_ban > self.so_luong(so_thung)[None, :]).mean(axis=0)

    # ---------------------------------------------------------------- sửa nghiệm vi phạm ràng buộc
    def sua_nghiem_co_ty_le(self, so_thung):
        """Co đồng loạt số thùng theo cùng một tỉ lệ cho đến khi vừa ràng buộc. Dùng được cho 1 phương án
        (vector) hoặc cả quần thể (ma trận). Đây cũng là cách "cắt theo tỉ lệ" của người nhập hàng thông thường."""
        g = np.asarray(so_thung)
        tien, kho = self.tien_nhap(g), self.the_tich_dung(g)
        he_so = np.minimum(1.0, np.minimum(self.ngan_sach / np.maximum(tien, 1e-12),
                                           self.suc_chua / np.maximum(kho, 1e-12)))
        if g.ndim == 2:
            he_so = he_so[:, None]
        return np.floor(g * he_so).astype(int)

    def sua_nghiem_tham_lam(self, so_thung):
        """Bớt từng thùng ở nơi "rẻ nhất": chi phí tăng ít nhất trên mỗi đơn vị nguồn lực giải phóng."""
        g = np.array(so_thung, dtype=int)
        while not self.hop_le(g):
            ung_vien = np.where(g > 0)[0]
            if len(ung_vien) == 0:
                break
            chi_phi_tang = self.bang_chi_phi[ung_vien, g[ung_vien] - 1] - self.bang_chi_phi[ung_vien, g[ung_vien]]
            giai_phong = (self.tien_moi_thung[ung_vien] / self.ngan_sach
                          + self.kho_moi_thung[ung_vien] / self.suc_chua)
            g[ung_vien[np.argmin(chi_phi_tang / giai_phong)]] -= 1
        return g

    # ---------------------------------------------------------------- chi phí THỰC TẾ khi biết nhu cầu thật
    def chi_phi_thuc_te(self, so_thung, nhu_cau_that):
        """Đánh giá một phương án với nhu cầu thật (dùng cho backtest). Trả về dict các khoản chi phí."""
        q = self.so_luong(so_thung)
        d = np.asarray(nhu_cau_that, float)
        nhap = float((self.gia_nhap * q).sum() + (self.phi_co_dinh * (q > 0)).sum())
        ton_kho = float((self.phi_ton_kho * np.maximum(q - d, 0)).sum())
        thieu = float((self.gia_ban * np.maximum(d - q, 0)).sum())
        return {
            "tong_chi_phi": nhap + ton_kho + thieu, "chi_phi_nhap": nhap, "chi_phi_ton_kho": ton_kho,
            "chi_phi_thieu": thieu, "ti_le_dap_ung": float(np.minimum(q, d).sum() / max(d.sum(), 1e-9)),
            "so_mat_hang_thieu": int((d > q).sum()),
        }


# -------------------------------------------------------------------- tạo bài toán từ dữ liệu
def tao_thong_so_mat_hang(gia_ban, nhu_cau_tb, hat=cau_hinh.HAT_NGAU_NHIEN):
    """Sinh các tham số GIẢ ĐỊNH (giá nhập, thể tích, phí...) mà dữ liệu bán hàng không có. Cố định bằng `hat`."""
    rng = np.random.default_rng(hat)
    n = len(gia_ban)
    gia_ban = np.asarray(gia_ban, float)
    gia_nhap = gia_ban * rng.uniform(*cau_hinh.TI_LE_GIA_VON, n)
    the_tich = rng.uniform(*cau_hinh.THE_TICH_MOI_DON_VI, n)
    phi_ton_kho = cau_hinh.TI_LE_PHI_TON_KHO * gia_nhap
    phi_co_dinh = cau_hinh.HE_SO_PHI_CO_DINH * gia_nhap if cau_hinh.DUNG_PHI_CO_DINH else np.zeros(n)
    quy_cach = np.ones(n, dtype=int)
    if cau_hinh.DUNG_QUY_CACH_THUNG:
        for i in range(n):  # mặt hàng bán ít thì nhập lẻ; bán nhiều mới nhập theo thùng
            cac_lua_chon = [k for k in cau_hinh.CAC_QUY_CACH_THUNG if k <= max(1.0, nhu_cau_tb[i])]
            quy_cach[i] = rng.choice(cac_lua_chon)
    return {"gia_ban": gia_ban, "gia_nhap": gia_nhap, "the_tich": the_tich,
            "phi_ton_kho": phi_ton_kho, "phi_co_dinh": phi_co_dinh, "quy_cach": quy_cach}


def tao_bai_toan(thong_so, kich_ban, nhu_cau_tb, ti_le_ngan_sach=cau_hinh.TI_LE_NGAN_SACH,
                 ti_le_suc_chua=cau_hinh.TI_LE_SUC_CHUA):
    """Ngân sách B và sức chứa V được đặt bằng một TỈ LỆ của lượng cần để nhập đủ nhu cầu trung bình.
    Tỉ lệ càng nhỏ, ràng buộc càng chặt."""
    nhu_cau_tb = np.asarray(nhu_cau_tb, float)
    ngan_sach = ti_le_ngan_sach * float((thong_so["gia_nhap"] * nhu_cau_tb).sum())
    suc_chua = ti_le_suc_chua * float((thong_so["the_tich"] * nhu_cau_tb).sum())
    return BaiToan(**thong_so, kich_ban=kich_ban, ngan_sach=ngan_sach, suc_chua=suc_chua)
