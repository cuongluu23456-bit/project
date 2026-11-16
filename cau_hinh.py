"""Cấu hình trung tâm: mọi tham số GIẢ ĐỊNH của dự án nằm ở đây.

Dữ liệu UCI Online Retail KHÔNG có giá vốn, thể tích kho, phí tồn kho... nên các giá trị đó
là giả định của nhóm. Hãy đọc file này đầu tiên và ghi các giả định vào báo cáo.
"""
from pathlib import Path

THU_MUC_GOC = Path(__file__).resolve().parent
DUONG_DAN_DU_LIEU = THU_MUC_GOC / "data" / "ban_hang_theo_ngay.csv"  # file thật (xem README)
THU_MUC_KET_QUA = THU_MUC_GOC / "ket_qua"

HAT_NGAU_NHIEN = 42  # cố định để kết quả lặp lại được

# ---------------------------------------------------------------- Dữ liệu
SO_MAT_HANG = 20

# ---------------------------------------------------------------- Giả định về mặt hàng
TI_LE_GIA_VON = (0.35, 0.55)       # giá nhập = 35%–55% giá bán (mỗi mặt hàng một giá trị ngẫu nhiên)
THE_TICH_MOI_DON_VI = (0.5, 3.0)   # thể tích kho chiếm bởi mỗi đơn vị hàng
TI_LE_PHI_TON_KHO = 0.03           # phí tồn kho cho mỗi đơn vị dư = 3% giá nhập

# Hai yếu tố "khó" làm bài toán không còn lồi (Greedy dễ bị mắc kẹt, GA và DP xử lý được).
# Đặt False để quay về bài toán đơn giản.
DUNG_PHI_CO_DINH = True            # mỗi lần nhập một mặt hàng mất thêm phí cố định K_i
HE_SO_PHI_CO_DINH = 4              # K_i = 4 × giá nhập (tương đương giá trị 4 đơn vị hàng)
DUNG_QUY_CACH_THUNG = True         # chỉ được nhập theo thùng (bội số của quy cách)
CAC_QUY_CACH_THUNG = (1, 6, 12)    # mặt hàng bán ít sẽ nhập lẻ từng cái, hàng bán nhiều nhập theo thùng

# ---------------------------------------------------------------- Dự báo
TI_LE_TRAIN = 0.6                  # 60% số ngày đầu để huấn luyện
TI_LE_VALID = 0.2                  # 20% tiếp theo để chọn mô hình và đo sai số (kịch bản nhu cầu)
                                   # 20% cuối cùng là tập test để backtest
SO_KICH_BAN = 200                  # số kịch bản nhu cầu sinh ra cho mỗi ngày

# ---------------------------------------------------------------- Ràng buộc mặc định
TI_LE_NGAN_SACH = 0.8              # ngân sách B = 80% số tiền cần để nhập đủ nhu cầu trung bình
TI_LE_SUC_CHUA = 0.8               # sức chứa kho V = 80% thể tích cần cho nhu cầu trung bình
NGUONG_NGUY_CO_THIEU = 0.8         # cảnh báo nếu xác suất thiếu hàng >= 80% (ngưỡng thấp hơn sẽ cảnh báo gần như mọi mặt hàng
                                   # vì ngân sách/kho đang chặt và giá bán cao hơn giá nhập khiến mức nhập tối ưu vốn đã có ~40–50% khả năng thiếu)

# ---------------------------------------------------------------- Thuật toán
LUOI_DP_NGAN_SACH = 500            # DP chia ngân sách thành 500 ô (lưới càng mịn càng chính xác, càng chậm)
LUOI_DP_KHO = 300                  # DP chia sức chứa kho thành 300 ô

GA_CACH_RANG_BUOC = "sua"          # "sua" (co về mức hợp lệ) hoặc "phat" (cộng phạt); xem thực nghiệm E2
GA_KICH_THUOC_QUAN_THE = 150
GA_SO_THE_HE = 400
GA_XAC_SUAT_LAI = 0.9
GA_SO_UU_TU = 2                    # số cá thể tốt nhất được giữ nguyên qua mỗi thế hệ
GA_KICH_THUOC_GIAI_DAU = 3         # chọn lọc giải đấu: bốc 3 cá thể, lấy cá thể tốt nhất
