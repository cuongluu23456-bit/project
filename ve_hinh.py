"""Vẽ các biểu đồ cho báo cáo (matplotlib). Mỗi hàm lưu một file PNG.

Màu: mỗi phương pháp có MỘT màu cố định ở mọi biểu đồ (bảng màu đã kiểm tra khả năng phân biệt cho người
mù màu). Chữ luôn dùng màu mực, không dùng màu của chuỗi dữ liệu. Các màu nhạt có nhãn giá trị đi kèm.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import du_bao  # noqa: E402
from thuc_nghiem import TEN_DP, TEN_GA, TEN_HOAN_HAO, TEN_THAM_LAM, TEN_THONG_THUONG  # noqa: E402

MAU = {TEN_THONG_THUONG: "#2a78d6", TEN_THAM_LAM: "#eb6834", TEN_GA: "#1baf7a",
       TEN_DP: "#eda100", TEN_HOAN_HAO: "#e87ba4"}
KIEU_DIEM = {TEN_THONG_THUONG: "o", TEN_THAM_LAM: "s", TEN_GA: "^", TEN_DP: "D"}
MAU_DU_BAO = "#6250d6"
NEN, MUC, MUC_PHU, MUC_NHAT, LUOI, TRUC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _dinh_dang(ax, luoi="y"):
    ax.set_facecolor(NEN)
    for canh in ("top", "right"):
        ax.spines[canh].set_visible(False)
    for canh in ("left", "bottom"):
        ax.spines[canh].set_color(TRUC)
    ax.tick_params(colors=MUC_PHU, labelsize=9)
    ax.grid(False)
    if luoi:
        ax.grid(True, axis=luoi, color=LUOI, linewidth=0.8)
    ax.set_axisbelow(True)


def _luu(fig, duong_dan, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(duong_dan, facecolor=NEN, dpi=150)
    plt.close(fig)


def ve_du_bao(goi, duong_dan):
    """Dự báo so với thực tế trên tập test, cho một mặt hàng bán nhiều và một mặt hàng trung bình."""
    bang = goi.test.assign(du_bao=goi.mo_hinh.predict(goi.test[du_bao.COT_DAC_TRUNG]))
    trung_binh = bang.groupby("ma_hang_so")["so_luong"].mean().sort_values()
    cac_chon = [trung_binh.index[-1], trung_binh.index[len(trung_binh) // 2]]
    fig, cac_truc = plt.subplots(2, 1, figsize=(9, 6.2), facecolor=NEN, sharex=True)
    for ax, ma_so in zip(cac_truc, cac_chon):
        b = bang[bang["ma_hang_so"] == ma_so].sort_values("ngay")
        _dinh_dang(ax)
        ax.plot(b["ngay"], b["so_luong"], color=MUC_PHU, lw=1.1, label="Thực tế")
        ax.plot(b["ngay"], b["du_bao"], color=MAU_DU_BAO, lw=1.6, label=f"Dự báo ({goi.ten_mo_hinh})")
        wape = 100 * np.abs(b["so_luong"] - b["du_bao"]).sum() / max(b["so_luong"].sum(), 1e-9)
        ax.set_title(f"{b['ten_hang'].iloc[0]} ({b['ma_hang'].iloc[0]}) – WAPE {wape:.0f}%",
                     color=MUC, loc="left", fontsize=10)
        ax.set_ylabel("Số lượng bán/ngày", color=MUC_PHU)
        ax.legend(frameon=False, fontsize=9, labelcolor=MUC_PHU, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    cac_truc[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.suptitle("Dự báo nhu cầu so với thực tế (tập test)", color=MUC, x=0.01, ha="left", fontsize=11)
    _luu(fig, duong_dan, rect=(0, 0, 1, 0.96))


def ve_ke_hoach_ngay_mai(ke_hoach, duong_dan):
    """Số lượng nhập ngày mai của từng mặt hàng theo từng phương pháp (mỗi phương pháp một dấu khác nhau)."""
    bang = ke_hoach["bang"]
    cac_ten = [TEN_THONG_THUONG, TEN_THAM_LAM, TEN_GA, TEN_DP]
    y = np.arange(len(bang))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=NEN)
    _dinh_dang(ax, luoi="x")
    ax.scatter(bang["Dự báo"], y, marker="|", s=260, color=MUC_NHAT, linewidths=1.6, label="Dự báo nhu cầu", zorder=2)
    for j, ten in enumerate(cac_ten):
        lech = (j - 1.5) * 0.2
        ax.scatter(bang[ten], y + lech, marker=KIEU_DIEM[ten], s=34, color=MAU[ten], edgecolors=NEN,
                   linewidths=0.8, label=ten, zorder=3)
    ax.set_yticks(y, bang["Mã hàng"])
    ax.set_xlabel("Số lượng nhập (đơn vị)", color=MUC_PHU)
    fig.suptitle(f"Kế hoạch nhập hàng ngày {ke_hoach['ngay']:%d/%m/%Y}  (phương án đề xuất: {ke_hoach['de_xuat']})",
                 color=MUC, x=0.01, ha="left", fontsize=11)
    cac_dau, cac_nhan = ax.get_legend_handles_labels()
    fig.legend(cac_dau, cac_nhan, frameon=False, fontsize=9, labelcolor=MUC_PHU, ncol=5, loc="upper left",
               bbox_to_anchor=(0.01, 0.965), handletextpad=0.3, columnspacing=1.2)
    _luu(fig, duong_dan, rect=(0, 0, 1, 0.945))


def ve_chi_phi_e2(bang, duong_dan):
    """E2: chi phí kỳ vọng cao hơn DP bao nhiêu % ở từng mức ràng buộc (thấp hơn = tốt hơn)."""
    cac_ti_le = sorted(bang["Tỉ lệ ràng buộc"].unique())
    cac_nhom = [("thong_thuong", TEN_THONG_THUONG), ("tham_lam", TEN_THAM_LAM), ("ga", TEN_GA)]
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=NEN)
    _dinh_dang(ax)
    rong, nhom_rong = 0.24, 0.8
    for j, (nhom, ten) in enumerate(cac_nhom):
        gia_tri, sai_so = [], []
        for ti_le in cac_ti_le:
            dong = bang[(bang["Tỉ lệ ràng buộc"] == ti_le) & (bang["nhom"] == nhom)]
            dong = dong.loc[[dong["Chi phí kỳ vọng"].idxmin()]] if nhom == "tham_lam" else dong  # Tham lam: cách tốt nhất
            gia_tri.append(float(dong["Hơn DP (%)"].iloc[0]))
            chi_phi_dp = float(bang[(bang["Tỉ lệ ràng buộc"] == ti_le) & (bang["nhom"] == "dp")]["Chi phí kỳ vọng"].iloc[0])
            sai_so.append(100 * float(dong["± Độ lệch chuẩn"].iloc[0]) / chi_phi_dp)
        x = np.arange(len(cac_ti_le)) + (j - 1) * (rong + 0.03)
        ax.bar(x, gia_tri, width=rong, color=MAU[ten], label=ten, edgecolor=NEN, linewidth=1.5, zorder=3)
        if nhom == "ga":
            ax.errorbar(x, gia_tri, yerr=sai_so, fmt="none", ecolor=MUC_PHU, capsize=3, lw=1, zorder=4)
        for xi, gt, ss in zip(x, gia_tri, sai_so):
            ax.text(xi, gt + (ss if gt >= 0 else -ss) + (0.25 if gt >= 0 else -0.25),
                    f"{gt:+.2f}%" if abs(gt) < 1 else f"{gt:+.1f}%", ha="center",
                    va="bottom" if gt >= 0 else "top", fontsize=8.5, color=MUC)
    ax.axhline(0, color=MUC_NHAT, lw=1.2, zorder=2)
    ax.set_xticks(np.arange(len(cac_ti_le)), [f"Ngân sách & kho = {int(100 * t)}%\nmức cần" for t in cac_ti_le])
    ax.set_ylabel("Chi phí kỳ vọng cao hơn DP (%)\n(DP là đường 0)", color=MUC_PHU)
    ax.set_title("E2 – Chi phí kỳ vọng so với DP (thấp hơn = tốt hơn; GA: trung bình ± độ lệch chuẩn)",
                 color=MUC, loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUC_PHU, loc="upper left")
    _luu(fig, duong_dan)


def ve_hoi_tu_ga(lich_su_ga, duong_dan):
    """Đường hội tụ của GA (trung bình ± độ lệch chuẩn qua các seed) so với các mốc ngang."""
    ls = lich_su_ga["lich_su"]
    the_he = np.arange(1, ls.shape[1] + 1)
    trung_binh, do_lech = ls.mean(axis=0), ls.std(axis=0)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=NEN)
    _dinh_dang(ax)
    ax.fill_between(the_he, trung_binh - do_lech, trung_binh + do_lech, color=MAU[TEN_GA], alpha=0.18, linewidth=0)
    ax.plot(the_he, trung_binh, color=MAU[TEN_GA], lw=1.8)
    cac_moc = [(lich_su_ga["chi_phi_thong_thuong"], TEN_THONG_THUONG), (lich_su_ga["chi_phi_tham_lam"], TEN_THAM_LAM),
               (lich_su_ga["chi_phi_dp"], TEN_DP)]
    for gia_tri, ten in cac_moc:
        ax.axhline(gia_tri, color=MAU[ten], lw=1.4, ls="--")
    thap = min(trung_binh.min(), lich_su_ga["chi_phi_dp"])
    cao = max(lich_su_ga["chi_phi_thong_thuong"], trung_binh[min(20, len(trung_binh) - 1)])
    ax.set_ylim(thap - 0.08 * (cao - thap), cao + 0.12 * (cao - thap))
    nhan_x = the_he[-1]
    # DP và GA gần như trùng nhau: nhãn DP đặt phía trên đường, nhãn GA đặt phía dưới để không đè lên nhau
    cac_nhan = [(lich_su_ga["chi_phi_thong_thuong"], TEN_THONG_THUONG, "bottom"),
                (lich_su_ga["chi_phi_tham_lam"], TEN_THAM_LAM, "bottom"),
                (lich_su_ga["chi_phi_dp"], TEN_DP, "bottom"),
                (trung_binh[-1], f"{TEN_GA} (trung bình)", "top")]
    for gia_tri, ten, vi_tri in cac_nhan:
        ax.text(nhan_x, gia_tri, f" {ten}: {gia_tri:,.1f} ", va=vi_tri, ha="right", fontsize=8.5, color=MUC)
    ax.set_xlabel("Thế hệ", color=MUC_PHU)
    ax.set_ylabel("Chi phí kỳ vọng tốt nhất của quần thể", color=MUC_PHU)
    ax.set_title(f"GA hội tụ về gần DP (ràng buộc = {int(100 * lich_su_ga['ti_le'])}% mức cần, {ls.shape[0]} seed)",
                 color=MUC, loc="left", fontsize=10.5)
    _luu(fig, duong_dan)


def ve_backtest(tom_tat, duong_dan):
    """E3: chi phí thực tế trung bình mỗi ngày và tỉ lệ đáp ứng nhu cầu của từng phương pháp."""
    cac_ten = list(tom_tat["Phương pháp"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=NEN)
    for ax, cot, tieu_de, dinh_dang in ((ax1, "Chi phí TB/ngày", "Chi phí thực tế trung bình mỗi ngày", "{:,.0f}"),
                                        (ax2, "Tỉ lệ đáp ứng (%)", "Tỉ lệ nhu cầu được đáp ứng (%)", "{:.1f}%")):
        _dinh_dang(ax)
        gia_tri = tom_tat[cot].to_numpy()
        ax.bar(np.arange(len(cac_ten)), gia_tri, color=[MAU[t] for t in cac_ten], width=0.7, edgecolor=NEN,
               linewidth=1.5, zorder=3)
        for i, gt in enumerate(gia_tri):
            ax.text(i, gt, dinh_dang.format(gt), ha="center", va="bottom", fontsize=9, color=MUC)
        ax.set_xticks(np.arange(len(cac_ten)), [t.replace(" (", "\n(").replace("Nhập ", "Nhập\n").replace("Dự báo ", "Dự báo\n")
                                                for t in cac_ten], fontsize=8.5)
        ax.set_title(tieu_de, color=MUC, loc="left", fontsize=10.5)
        ax.set_ylim(0, gia_tri.max() * 1.12)
    _luu(fig, duong_dan)
